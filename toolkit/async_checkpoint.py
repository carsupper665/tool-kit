"""
Background checkpoint writer.

The training thread captures an immutable CPU snapshot at a safe optimizer boundary and
hands it to a single background writer, which serializes it, stages it, verifies it and
publishes it atomically. A checkpoint group (for example the Wan 2.2 high + low noise LoRA
pair, plus the optimizer state for a recovery save) is only considered valid once its
manifest under `<save_root>/.checkpoints/<group>.json` has `status: completed`.

Publish order for a group:
    1. write every file into `.checkpoints/.staging/<group>/`, fsync, verify
    2. write the manifest with `status: pending` (marks the final paths as uncommitted)
    3. os.replace() every staged file into its final path (same filesystem)
    4. rewrite the manifest with `status: completed`

Anything that is not referenced by a completed manifest is ignored by resume, and a
pending manifest left behind by a crash is rolled back by `recover_incomplete()`.

This module only touches CPU tensors and plain data. It never reads the live network or
optimizer, so nothing here may receive a reference to them.
"""

import atexit
import json
import os
import shutil
import threading
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_DIR = ".checkpoints"
STAGING_DIR = ".staging"

CHECKPOINT_TYPE_PREVIEW = "preview"
CHECKPOINT_TYPE_RECOVERY = "recovery"

STATUS_PENDING = "pending"
STATUS_DELETING = "deleting"
STATUS_COMPLETED = "completed"

ROLE_LORA = "lora"
ROLE_OPTIMIZER = "optimizer"

# outcome recorded in the timing log
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"


class AsyncCheckpointError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# snapshot helpers (called on the training thread)
# ---------------------------------------------------------------------------

def snapshot_to_cpu(obj: Any) -> Any:
    """Deep copy a (nested) state dict, giving every tensor its own CPU storage."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, snapshot_to_cpu(v)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: snapshot_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [snapshot_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(snapshot_to_cpu(v) for v in obj)
    # ints, floats, strings, None, and anything exotic an optimizer may store
    import copy
    return copy.deepcopy(obj)


def estimate_nbytes(obj: Any, dtype: Optional[torch.dtype] = None) -> int:
    """Bytes a snapshot of `obj` will take, without copying anything.
    If dtype is given, floating point tensors are counted as that dtype."""
    if isinstance(obj, torch.Tensor):
        if dtype is not None and obj.is_floating_point():
            return obj.numel() * torch.empty((), dtype=dtype).element_size()
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(estimate_nbytes(v, dtype) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_nbytes(v, dtype) for v in obj)
    return 0


def get_available_ram_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def resolve_max_pending_bytes(max_pending_bytes: Optional[int]) -> int:
    """`None` means: a quarter of the RAM that is available right now. Never unbounded."""
    if max_pending_bytes is not None:
        return int(max_pending_bytes)
    available = get_available_ram_bytes()
    if available is None:
        # unknown platform, fall back to a conservative fixed budget
        return 8 * 1024 ** 3
    return max(available // 4, 1)


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------

@dataclass
class CheckpointFile:
    # path relative to save_root where the file is published
    rel_path: str
    role: str = ROLE_LORA
    # safetensors payload
    tensors: Optional[Dict[str, torch.Tensor]] = None
    # torch.save payload (optimizer state)
    obj: Any = None


@dataclass
class CheckpointRequest:
    group: str
    step: int
    checkpoint_type: str
    files: List[CheckpointFile]
    # metadata written into every LoRA safetensors file of the group
    lora_metadata: Optional[Dict[str, str]] = None
    # full (unsplit) LoRA state dict the model hash is computed over, so split files
    # carry the same hash metadata as the synchronous save path writes
    hash_source: Optional[Dict[str, torch.Tensor]] = None
    epoch: Optional[int] = None
    job_name: Optional[str] = None
    model_arch: Optional[str] = None
    nbytes: int = 0
    # timings measured on the training thread, merged into the timing log
    producer_timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------------

def checkpoint_dir(save_root: str) -> str:
    return os.path.join(save_root, CHECKPOINT_DIR)


def manifest_path(save_root: str, group: str) -> str:
    return os.path.join(checkpoint_dir(save_root), f"{group}.json")


def _fsync_file(path: str):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str):
    # directory fsync is not supported everywhere (windows), it is best effort
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: str, data: dict, fsync: bool):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp_path, path)
    if fsync:
        _fsync_dir(os.path.dirname(path))


def _remove_quiet(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        pass


def read_manifest(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "group" not in data or "files" not in data:
        return None
    return data


def list_manifests(save_root: str, status: Optional[str] = STATUS_COMPLETED) -> List[dict]:
    """Manifests in save_root, oldest commit first."""
    root = checkpoint_dir(save_root)
    if not os.path.isdir(root):
        return []
    manifests = []
    for name in os.listdir(root):
        if not name.endswith(".json"):
            continue
        data = read_manifest(os.path.join(root, name))
        if data is None:
            continue
        if status is not None and data.get("status") != status:
            continue
        manifests.append(data)
    manifests.sort(key=lambda m: (m.get("commit_seq", 0), m.get("step", 0)))
    return manifests


def manifest_files(manifest: dict, role: Optional[str] = None) -> List[dict]:
    return [f for f in manifest.get("files", []) if role is None or f.get("role") == role]


def manifest_is_intact(save_root: str, manifest: dict) -> bool:
    """All files the manifest lists exist with the recorded size."""
    for entry in manifest.get("files", []):
        path = os.path.join(save_root, entry["path"])
        try:
            if os.path.getsize(path) != entry.get("size"):
                return False
        except OSError:
            return False
    return True


def find_latest_recovery(save_root: str) -> Optional[dict]:
    """Highest step completed group that has LoRA weights and optimizer state, all present."""
    candidates = [
        m for m in list_manifests(save_root)
        if m.get("checkpoint_type") == CHECKPOINT_TYPE_RECOVERY
        and manifest_files(m, ROLE_LORA)
        and manifest_files(m, ROLE_OPTIMIZER)
    ]
    candidates.sort(key=lambda m: (m.get("step", 0), m.get("commit_seq", 0)), reverse=True)
    for m in candidates:
        if manifest_is_intact(save_root, m):
            return m
    return None


def find_latest_lora(save_root: str) -> Optional[dict]:
    """Highest step completed preview or recovery group with intact LoRA files."""
    candidates = [
        m for m in list_manifests(save_root)
        if m.get("checkpoint_type") in (CHECKPOINT_TYPE_PREVIEW, CHECKPOINT_TYPE_RECOVERY)
        and manifest_files(m, ROLE_LORA)
    ]
    candidates.sort(key=lambda m: (m.get("step", 0), m.get("commit_seq", 0)), reverse=True)
    for m in candidates:
        lora_only = dict(m, files=manifest_files(m, ROLE_LORA))
        if manifest_is_intact(save_root, lora_only):
            return m
    return None


def recover_incomplete(save_root: str, log: Callable[[str], None] = print) -> List[str]:
    """Roll back groups a crash left behind: anything listed by a non completed manifest
    is deleted, as are staging leftovers. Returns the groups that were rolled back."""
    root = checkpoint_dir(save_root)
    if not os.path.isdir(root):
        return []
    rolled_back = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.endswith(".json.tmp"):
            _remove_quiet(path)
            continue
        if not name.endswith(".json"):
            continue
        data = read_manifest(path)
        if data is None:
            log(f"Async checkpoint: ignoring unreadable manifest {path}")
            continue
        if data.get("status") == STATUS_COMPLETED:
            continue
        for entry in data.get("files", []):
            _remove_quiet(os.path.join(save_root, entry["path"]))
        _remove_quiet(path)
        _remove_empty_group_dir(save_root, data["group"])
        rolled_back.append(data["group"])
        log(f"Async checkpoint: rolled back incomplete checkpoint {data['group']}")
    _remove_quiet(os.path.join(root, STAGING_DIR))
    return rolled_back


def _remove_empty_group_dir(save_root: str, group: str):
    group_dir = os.path.join(checkpoint_dir(save_root), group)
    try:
        if os.path.isdir(group_dir) and not os.listdir(group_dir):
            os.rmdir(group_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# writer side
# ---------------------------------------------------------------------------

def _verify_safetensors(path: str, expected_keys: int):
    with safe_open(path, framework="pt") as f:
        found = len(list(f.keys()))
    if found != expected_keys:
        raise AsyncCheckpointError(
            f"verification failed for {path}: expected {expected_keys} tensors, found {found}"
        )


def commit_request(
    save_root: str,
    request: CheckpointRequest,
    commit_seq: int,
    metadata_hasher: Optional[Callable[[Dict[str, torch.Tensor], OrderedDict], OrderedDict]] = None,
    fsync: bool = True,
) -> Dict[str, float]:
    """Serialize, stage, verify and publish one checkpoint group. Returns timings in ms.
    On failure everything this call created is removed and the exception propagates."""
    timings: Dict[str, float] = {}
    root = checkpoint_dir(save_root)
    staging = os.path.join(root, STAGING_DIR, request.group)
    _remove_quiet(staging)
    os.makedirs(staging, exist_ok=True)

    published: List[str] = []
    pending_manifest_written = False
    mpath = manifest_path(save_root, request.group)
    try:
        t = time.perf_counter()
        lora_metadata = request.lora_metadata
        if lora_metadata is not None and len(lora_metadata) == 0:
            lora_metadata = None
        if lora_metadata is None:
            lora_metadata = OrderedDict()
        else:
            lora_metadata = OrderedDict(lora_metadata)
        if metadata_hasher is not None and request.hash_source is not None:
            lora_metadata = metadata_hasher(request.hash_source, lora_metadata)
        timings["convert_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        staged = []
        for i, item in enumerate(request.files):
            staged_path = os.path.join(staging, f"{i}_{os.path.basename(item.rel_path)}")
            if item.tensors is not None:
                save_file(item.tensors, staged_path, metadata=lora_metadata if item.role == ROLE_LORA else None)
            else:
                torch.save(item.obj, staged_path)
            if fsync:
                _fsync_file(staged_path)
            staged.append(staged_path)
        timings["disk_write_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        entries = []
        for item, staged_path in zip(request.files, staged):
            size = os.path.getsize(staged_path)
            if size == 0:
                raise AsyncCheckpointError(f"verification failed for {staged_path}: empty file")
            if item.tensors is not None:
                _verify_safetensors(staged_path, len(item.tensors))
            entries.append({"path": item.rel_path.replace(os.sep, "/"), "role": item.role, "size": size})
        timings["verify_ms"] = (time.perf_counter() - t) * 1000

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "group": request.group,
            "job_name": request.job_name,
            "step": request.step,
            "epoch": request.epoch,
            "checkpoint_type": request.checkpoint_type,
            "model_arch": request.model_arch,
            "commit_seq": commit_seq,
            "status": STATUS_PENDING,
            "files": entries,
        }

        t = time.perf_counter()
        _write_json_atomic(mpath, manifest, fsync)
        pending_manifest_written = True
        for item, staged_path in zip(request.files, staged):
            final_path = os.path.join(save_root, item.rel_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.replace(staged_path, final_path)
            published.append(final_path)
        if fsync:
            for d in {os.path.dirname(p) for p in published}:
                _fsync_dir(d)
        manifest["status"] = STATUS_COMPLETED
        manifest["committed_at"] = time.time()
        _write_json_atomic(mpath, manifest, fsync)
        timings["publish_ms"] = (time.perf_counter() - t) * 1000
    except BaseException:
        # nothing of this group may stay visible
        if pending_manifest_written:
            for path in published:
                _remove_quiet(path)
            _remove_quiet(mpath)
            _remove_empty_group_dir(save_root, request.group)
        raise
    finally:
        _remove_quiet(staging)
    return timings


def apply_retention(
    save_root: str,
    keep_lora_groups: int,
    keep_recovery_groups: int,
    fsync: bool = True,
    log: Callable[[str], None] = print,
) -> List[str]:
    """Group based cleanup. Keeps the `keep_lora_groups` newest preview/recovery groups,
    and the optimizer state (plus the LoRA files) of the `keep_recovery_groups` newest
    recovery groups. Newest = latest commit, so a
    timeline abandoned by a resume ages out instead of shadowing the new one."""
    managed = list_manifests(save_root)
    newest_first = list(reversed(managed))
    keep_lora = {m["group"] for m in newest_first[:max(keep_lora_groups, 0)]}
    recovery = [m for m in newest_first if manifest_files(m, ROLE_OPTIMIZER)]
    keep_recovery = {m["group"] for m in recovery[:max(keep_recovery_groups, 0)]}

    removed = []
    for m in managed:
        group = m["group"]
        mpath = manifest_path(save_root, group)
        if group not in keep_lora and group not in keep_recovery:
            # mark first, so a crash halfway is rolled back by recover_incomplete()
            _write_json_atomic(mpath, dict(m, status=STATUS_DELETING), fsync)
            for entry in m.get("files", []):
                _remove_quiet(os.path.join(save_root, entry["path"]))
            _remove_quiet(mpath)
            _remove_empty_group_dir(save_root, group)
            removed.append(group)
            log(f"Removing old checkpoint group: {group}")
        elif group not in keep_recovery and manifest_files(m, ROLE_OPTIMIZER):
            # keep the LoRA preview, drop the optimizer state
            optimizer_files = manifest_files(m, ROLE_OPTIMIZER)
            trimmed = dict(
                m,
                checkpoint_type=CHECKPOINT_TYPE_PREVIEW,
                files=manifest_files(m, ROLE_LORA),
            )
            _write_json_atomic(mpath, trimmed, fsync)
            for entry in optimizer_files:
                _remove_quiet(os.path.join(save_root, entry["path"]))
            _remove_empty_group_dir(save_root, group)
            log(f"Removing old recovery state: {group}")
    return removed


class AsyncCheckpointManager:
    """Bounded queue in front of a single background writer.

    `max_pending_saves` counts queued *and* in-flight requests. `max_pending_bytes` caps
    the CPU snapshot memory held by those requests. A reservation must be taken before the
    snapshot is allocated, so backpressure stops the producer before RAM is spent.
    """

    def __init__(
        self,
        save_root: str,
        max_pending_saves: int = 2,
        max_pending_bytes: Optional[int] = None,
        keep_lora_groups: int = 5,
        keep_recovery_groups: int = 3,
        metadata_hasher: Optional[Callable] = None,
        fsync: bool = True,
        flush_on_exit: bool = True,
        timings_path: Optional[str] = None,
        log: Callable[[str], None] = print,
    ):
        if max_pending_saves < 1:
            raise ValueError("max_pending_saves must be at least 1")
        self.save_root = save_root
        self.max_pending_saves = max_pending_saves
        self.max_pending_bytes = resolve_max_pending_bytes(max_pending_bytes)
        self.keep_lora_groups = keep_lora_groups
        self.keep_recovery_groups = keep_recovery_groups
        self.metadata_hasher = metadata_hasher
        self.fsync = fsync
        self.timings_path = timings_path
        self.log = log

        self._cond = threading.Condition()
        self._queue: deque = deque()
        self._pending_count = 0
        self._pending_bytes = 0
        self._error: Optional[BaseException] = None
        self._closed = False

        existing = list_manifests(save_root, status=None)
        self._commit_seq = max([m.get("commit_seq", 0) for m in existing] + [0])

        os.makedirs(checkpoint_dir(save_root), exist_ok=True)
        self._worker = threading.Thread(target=self._run, name="aitk-checkpoint-writer", daemon=True)
        self._worker.start()
        self._atexit = None
        if flush_on_exit:
            self._atexit = self._close_at_exit
            atexit.register(self._atexit)

    # ---- producer side -------------------------------------------------

    def fits_budget(self, nbytes: int) -> bool:
        return nbytes <= self.max_pending_bytes

    def make_room(self, nbytes: int) -> bool:
        """Call before allocating a snapshot of `nbytes`.
        Returns True once room is reserved; hand the request to submit().
        A snapshot larger than max_pending_bytes can never be queued: then this waits until
        every pending save is written and its snapshot freed, so two snapshots are never
        held at once, and returns False; hand the request to write_now()."""
        if self.fits_budget(nbytes):
            self.reserve(nbytes)
            return True
        self.drain()
        return False

    def reserve(self, nbytes: int):
        """Wait for room for one more request of `nbytes`. Raises if the writer failed."""
        if not self.fits_budget(nbytes):
            raise AsyncCheckpointError(
                f"snapshot of {nbytes} bytes exceeds max_pending_bytes={self.max_pending_bytes}"
            )
        with self._cond:
            while True:
                self._raise_if_failed_locked()
                if self._closed:
                    raise AsyncCheckpointError("checkpoint manager is closed")
                has_room = (
                    self._pending_count < self.max_pending_saves
                    and self._pending_bytes + nbytes <= self.max_pending_bytes
                )
                if has_room:
                    self._pending_count += 1
                    self._pending_bytes += nbytes
                    return
                self._cond.wait()

    def release(self, nbytes: int):
        """Give back a reservation that will not be submitted."""
        with self._cond:
            self._pending_count -= 1
            self._pending_bytes -= nbytes
            self._cond.notify_all()

    def submit(self, request: CheckpointRequest):
        """Queue a request whose bytes were reserved with reserve(request.nbytes)."""
        with self._cond:
            self._queue.append(request)
            self._cond.notify_all()

    def drain(self):
        """Block until every submitted request is written. Raises if one failed."""
        with self._cond:
            while self._pending_count > 0 and self._error is None:
                self._cond.wait()
            self._raise_if_failed_locked()

    def raise_if_failed(self):
        with self._cond:
            self._raise_if_failed_locked()

    def close(self, drain: bool = True):
        if self._atexit is not None:
            atexit.unregister(self._atexit)
            self._atexit = None
        try:
            if drain:
                self.drain()
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()
            self._worker.join()

    def _close_at_exit(self):
        try:
            self.close(drain=True)
        except BaseException as e:
            self.log(f"Async checkpoint: error while flushing on exit: {e}")

    @property
    def pending_count(self) -> int:
        with self._cond:
            return self._pending_count

    @property
    def pending_bytes(self) -> int:
        with self._cond:
            return self._pending_bytes

    def _raise_if_failed_locked(self):
        if self._error is not None:
            raise AsyncCheckpointError(f"background checkpoint writer failed: {self._error}") from self._error

    # ---- writer side ---------------------------------------------------

    def write_now(self, request: CheckpointRequest):
        """Write a request on the calling thread, after everything queued before it.
        Used when a snapshot does not fit max_pending_bytes. Raises on failure."""
        self.drain()
        with self._cond:
            self._raise_if_failed_locked()
            self._pending_count += 1
            self._pending_bytes += request.nbytes
            self._commit_seq += 1
            seq = self._commit_seq
        self._process(request, seq)
        self.raise_if_failed()

    def _run(self):
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                request: CheckpointRequest = self._queue.popleft()
                self._commit_seq += 1
                seq = self._commit_seq
            self._process(request, seq)

    def _process(self, request: CheckpointRequest, seq: int):
        try:
            if self._error is not None:
                # a previous request failed; do not publish anything after it
                raise AsyncCheckpointError("skipped after an earlier writer failure")
            t = time.perf_counter()
            timings = commit_request(
                self.save_root,
                request,
                commit_seq=seq,
                metadata_hasher=self.metadata_hasher,
                fsync=self.fsync,
            )
            tc = time.perf_counter()
            apply_retention(
                self.save_root,
                self.keep_lora_groups,
                self.keep_recovery_groups,
                fsync=self.fsync,
                log=self.log,
            )
            timings["cleanup_ms"] = (time.perf_counter() - tc) * 1000
            timings["writer_total_ms"] = (time.perf_counter() - t) * 1000
            self.log(
                f"Async checkpoint committed: {request.group} "
                f"({request.checkpoint_type}, step {request.step}) in {timings['writer_total_ms']:.0f} ms"
            )
            self._write_timings(request, timings, STATE_COMPLETED)
        except BaseException as e:
            self.log(f"Async checkpoint FAILED: {request.group}: {e}")
            if self._error is None:
                traceback.print_exc()
            self._write_timings(request, {}, STATE_FAILED, error=str(e))
            with self._cond:
                if self._error is None:
                    self._error = e
        finally:
            # drop the snapshot before giving the bytes back
            request.files = []
            request.hash_source = None
            with self._cond:
                self._pending_count -= 1
                self._pending_bytes -= request.nbytes
                self._cond.notify_all()

    def _write_timings(self, request: CheckpointRequest, timings: Dict[str, float], state: str, error: str = None):
        if self.timings_path is None:
            return
        record = {
            "time": time.time(),
            "kind": "writer",
            "mode": "async",
            "group": request.group,
            "step": request.step,
            "checkpoint_type": request.checkpoint_type,
            "state": state,
            "bytes": request.nbytes,
        }
        record.update({k: round(v, 3) for k, v in request.producer_timings.items()})
        record.update({k: round(v, 3) for k, v in timings.items()})
        if error is not None:
            record["error"] = error
        append_timing_record(self.timings_path, record)


_timing_lock = threading.Lock()


def append_timing_record(path: str, record: dict):
    with _timing_lock:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass
