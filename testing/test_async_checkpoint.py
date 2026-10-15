"""
Tests for toolkit/async_checkpoint.py. CPU only, no model needed:

    python -m pytest testing/test_async_checkpoint.py
"""
import json
import os
import sys
import threading
from collections import OrderedDict

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit import async_checkpoint as ac  # noqa: E402


def fake_hasher(state_dict, meta):
    meta = OrderedDict(meta)
    meta["sshs_model_hash"] = str(sum(int(v.float().sum().item() * 1000) for v in state_dict.values()))
    return meta


def lora_state(seed=0, transformer_split=True):
    g = torch.Generator().manual_seed(seed)
    sd = OrderedDict()
    for t in (1, 2) if transformer_split else (None,):
        prefix = f"diffusion_model.transformer_{t}." if t else "diffusion_model."
        sd[prefix + "blocks.0.attn.lora_A.weight"] = torch.randn(4, 8, generator=g, dtype=torch.bfloat16)
        sd[prefix + "blocks.0.attn.lora_B.weight"] = torch.randn(8, 4, generator=g, dtype=torch.bfloat16)
    return sd


def wan_split(state_dict, output_path):
    # mirrors Wan2214bModel.split_lora_for_save with both experts trained
    high, low = {}, {}
    for key, value in state_dict.items():
        if ".transformer_1." in key:
            high[key.replace(".transformer_1.", ".")] = value
        elif ".transformer_2." in key:
            low[key.replace(".transformer_2.", ".")] = value
    return [
        (output_path.replace(".safetensors", "_high_noise.safetensors"), high),
        (output_path.replace(".safetensors", "_low_noise.safetensors"), low),
    ]


def make_request(save_root, step, checkpoint_type=ac.CHECKPOINT_TYPE_PREVIEW, optimizer=None, name="xian", seed=None):
    sd = lora_state(seed if seed is not None else step)
    group = f"{name}_{str(step).zfill(9)}"
    files = [
        ac.CheckpointFile(rel_path=os.path.relpath(path, save_root), tensors=tensors)
        for path, tensors in wan_split(sd, os.path.join(save_root, f"{group}.safetensors"))
    ]
    if optimizer is not None:
        files.append(ac.CheckpointFile(
            rel_path=os.path.join(ac.CHECKPOINT_DIR, group, "optimizer.pt"),
            role=ac.ROLE_OPTIMIZER,
            obj=optimizer,
        ))
    return ac.CheckpointRequest(
        group=group,
        step=step,
        checkpoint_type=checkpoint_type,
        files=files,
        lora_metadata=OrderedDict(ss_output_name=name),
        hash_source=sd,
        job_name=name,
        model_arch="wan22_14b",
        nbytes=ac.estimate_nbytes(sd),
    ), sd


def visible_safetensors(save_root):
    return sorted(f for f in os.listdir(save_root) if f.endswith(".safetensors"))


# ---------------------------------------------------------------------------


def test_snapshot_is_independent_of_live_tensors():
    live = {"state": {0: {"exp_avg": torch.ones(3)}}, "param_groups": [{"lr": 1e-4, "params": [0]}]}
    snap = ac.snapshot_to_cpu(live)
    live["state"][0]["exp_avg"].add_(5)
    live["param_groups"][0]["lr"] = 9.0
    assert torch.equal(snap["state"][0]["exp_avg"], torch.ones(3))
    assert snap["param_groups"][0]["lr"] == 1e-4
    assert snap["state"][0]["exp_avg"].untyped_storage().data_ptr() != live["state"][0]["exp_avg"].untyped_storage().data_ptr()


def test_estimate_nbytes_uses_save_dtype():
    sd = {"a": torch.zeros(10, dtype=torch.float32), "b": torch.zeros(4, dtype=torch.int64)}
    assert ac.estimate_nbytes(sd) == 10 * 4 + 4 * 8
    assert ac.estimate_nbytes(sd, torch.bfloat16) == 10 * 2 + 4 * 8


def test_commit_matches_synchronous_wan_save(tmp_path):
    save_root = str(tmp_path / "async")
    sync_root = tmp_path / "sync"
    os.makedirs(save_root)
    sync_root.mkdir()
    request, sd = make_request(save_root, 500)

    ac.commit_request(save_root, request, commit_seq=1, metadata_hasher=fake_hasher)

    # what Network.save_weights + Wan2214bModel.save_lora write synchronously
    meta = fake_hasher(sd, OrderedDict(ss_output_name="xian"))
    for path, tensors in wan_split(sd, str(sync_root / "xian_000000500.safetensors")):
        save_file(tensors, path, metadata=meta)

    assert visible_safetensors(save_root) == [
        "xian_000000500_high_noise.safetensors",
        "xian_000000500_low_noise.safetensors",
    ]
    for name in visible_safetensors(save_root):
        a = load_file(os.path.join(save_root, name))
        b = load_file(str(sync_root / name))
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].dtype == b[k].dtype
            assert torch.equal(a[k], b[k])
        with safe_open(os.path.join(save_root, name), "pt") as fa, safe_open(str(sync_root / name), "pt") as fb:
            assert fa.metadata() == fb.metadata()

    manifest = ac.read_manifest(ac.manifest_path(save_root, "xian_000000500"))
    assert manifest["status"] == ac.STATUS_COMPLETED
    assert manifest["step"] == 500
    assert [f["path"] for f in manifest["files"]] == visible_safetensors(save_root)
    assert not os.path.exists(os.path.join(save_root, ac.CHECKPOINT_DIR, ac.STAGING_DIR, "xian_000000500"))


def test_failed_write_publishes_nothing(tmp_path, monkeypatch):
    save_root = str(tmp_path)
    request, _ = make_request(save_root, 100)
    real_save_file = ac.save_file
    calls = []

    def flaky_save_file(tensors, path, metadata=None):
        calls.append(path)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        real_save_file(tensors, path, metadata=metadata)

    monkeypatch.setattr(ac, "save_file", flaky_save_file)
    with pytest.raises(OSError):
        ac.commit_request(save_root, request, commit_seq=1)

    assert visible_safetensors(save_root) == []
    assert ac.list_manifests(save_root, status=None) == []
    assert os.listdir(os.path.join(save_root, ac.CHECKPOINT_DIR, ac.STAGING_DIR)) == []


def test_failure_during_publish_rolls_back_moved_files(tmp_path, monkeypatch):
    save_root = str(tmp_path)
    request, _ = make_request(save_root, 100)
    real_replace = os.replace
    moved = []

    def flaky_replace(src, dst):
        if dst.endswith("_low_noise.safetensors"):
            raise OSError("simulated crash between the two experts")
        moved.append(dst)
        real_replace(src, dst)

    monkeypatch.setattr(ac.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        ac.commit_request(save_root, request, commit_seq=1)
    monkeypatch.setattr(ac.os, "replace", real_replace)

    # the pending manifest and the high noise expert were already published
    assert any(p.endswith("_high_noise.safetensors") for p in moved)
    assert visible_safetensors(save_root) == []
    assert ac.list_manifests(save_root, status=None) == []


def test_recover_incomplete_removes_half_published_group(tmp_path):
    save_root = str(tmp_path)
    done, _ = make_request(save_root, 100)
    ac.commit_request(save_root, done, commit_seq=1)

    # what a SIGKILL between publishing the high and the low noise file leaves behind
    pending, _ = make_request(save_root, 200)
    files = [{"path": f.rel_path, "role": f.role, "size": 1} for f in pending.files]
    with open(ac.manifest_path(save_root, pending.group), "w") as f:
        json.dump({"group": pending.group, "step": 200, "status": ac.STATUS_PENDING, "files": files}, f)
    save_file(pending.files[0].tensors, os.path.join(save_root, pending.files[0].rel_path))
    os.makedirs(os.path.join(save_root, ac.CHECKPOINT_DIR, ac.STAGING_DIR, pending.group))

    assert "xian_000000200_high_noise.safetensors" in visible_safetensors(save_root)
    assert ac.recover_incomplete(save_root, log=lambda *_: None) == [pending.group]
    assert visible_safetensors(save_root) == [
        "xian_000000100_high_noise.safetensors",
        "xian_000000100_low_noise.safetensors",
    ]
    assert [m["group"] for m in ac.list_manifests(save_root, status=None)] == [done.group]
    assert not os.path.exists(os.path.join(save_root, ac.CHECKPOINT_DIR, ac.STAGING_DIR))


def test_manager_writes_in_order_and_drains(tmp_path):
    save_root = str(tmp_path)
    manager = ac.AsyncCheckpointManager(
        save_root, max_pending_saves=2, max_pending_bytes=10 ** 9,
        keep_lora_groups=100, metadata_hasher=fake_hasher, fsync=False, log=lambda *_: None,
    )
    for step in range(100, 1100, 100):
        request, _ = make_request(save_root, step)
        manager.reserve(request.nbytes)
        manager.submit(request)
    manager.close()

    manifests = ac.list_manifests(save_root)
    assert [m["step"] for m in manifests] == list(range(100, 1100, 100))
    assert [m["commit_seq"] for m in manifests] == list(range(1, 11))
    assert all(ac.manifest_is_intact(save_root, m) for m in manifests)
    assert manager.pending_count == 0 and manager.pending_bytes == 0


class PausedWriter:
    """metadata_hasher that holds the background writer until release() is called"""

    def __init__(self):
        self.started = threading.Event()
        self.released = threading.Event()

    def __call__(self, state_dict, meta):
        self.started.set()
        assert self.released.wait(10)
        return meta

    def release(self):
        self.released.set()


def run_in_thread(fn):
    done = threading.Event()

    def target():
        fn()
        done.set()

    thread = threading.Thread(target=target)
    thread.start()
    return thread, done


def test_backpressure_blocks_until_writer_frees_room(tmp_path):
    save_root = str(tmp_path)
    writer = PausedWriter()
    manager = ac.AsyncCheckpointManager(
        save_root, max_pending_saves=1, max_pending_bytes=10 ** 9,
        metadata_hasher=writer, fsync=False, flush_on_exit=False, log=lambda *_: None,
    )
    first, _ = make_request(save_root, 100)
    manager.reserve(first.nbytes)
    manager.submit(first)
    assert writer.started.wait(10)

    # the in-flight save counts against max_pending_saves, so the producer waits
    second, _ = make_request(save_root, 200)
    thread, reserved = run_in_thread(lambda: manager.reserve(second.nbytes))
    assert not reserved.wait(0.3)
    writer.release()
    assert reserved.wait(10)
    thread.join()
    manager.submit(second)
    manager.close()
    assert [m["step"] for m in ac.list_manifests(save_root)] == [100, 200]


def test_byte_budget_blocks_until_bytes_are_released(tmp_path):
    manager = ac.AsyncCheckpointManager(
        str(tmp_path), max_pending_saves=4, max_pending_bytes=1000, fsync=False,
        flush_on_exit=False, log=lambda *_: None,
    )
    try:
        assert not manager.fits_budget(1001)
        with pytest.raises(ac.AsyncCheckpointError):
            manager.reserve(1001)
        manager.reserve(600)
        thread, reserved = run_in_thread(lambda: manager.reserve(600))
        assert not reserved.wait(0.3)
        manager.release(600)
        assert reserved.wait(10)
        thread.join()
        assert manager.pending_bytes == 600
        manager.release(600)
    finally:
        manager.close()


def test_write_now_for_oversized_snapshot(tmp_path):
    save_root = str(tmp_path)
    manager = ac.AsyncCheckpointManager(
        save_root, max_pending_bytes=1, fsync=False, flush_on_exit=False, log=lambda *_: None,
    )
    request, _ = make_request(save_root, 100)
    assert manager.make_room(request.nbytes) is False
    manager.write_now(request)
    manager.close()
    assert [m["step"] for m in ac.list_manifests(save_root)] == [100]


def test_oversized_snapshot_waits_for_pending_saves_before_allocating(tmp_path):
    # the old snapshot must be written and freed before the oversized one is allocated,
    # so the two are never held in RAM at the same time
    save_root = str(tmp_path)
    writer = PausedWriter()
    manager = ac.AsyncCheckpointManager(
        save_root, max_pending_bytes=10 ** 6, metadata_hasher=writer,
        fsync=False, flush_on_exit=False, log=lambda *_: None,
    )
    old, _ = make_request(save_root, 100)
    assert manager.make_room(old.nbytes) is True
    manager.submit(old)
    assert writer.started.wait(10)

    events = []

    def producer():
        queued = manager.make_room(10 ** 7)
        # stands in for get_state_dict() / snapshot_to_cpu()
        events.append(("snapshot", manager.pending_count, queued))

    thread, done = run_in_thread(producer)
    assert not done.wait(0.3)
    assert events == []
    writer.release()
    assert done.wait(10)
    thread.join()
    assert events == [("snapshot", 0, False)]
    assert [m["step"] for m in ac.list_manifests(save_root)] == [100]
    manager.close()


def test_writer_error_is_raised_to_the_producer(tmp_path):
    save_root = str(tmp_path)

    def broken_hasher(state_dict, meta):
        raise PermissionError("permission denied")

    manager = ac.AsyncCheckpointManager(
        save_root, metadata_hasher=broken_hasher, fsync=False, flush_on_exit=False, log=lambda *_: None,
    )
    request, _ = make_request(save_root, 100)
    manager.reserve(request.nbytes)
    manager.submit(request)
    with pytest.raises(ac.AsyncCheckpointError):
        manager.drain()
    with pytest.raises(ac.AsyncCheckpointError):
        manager.reserve(1)
    with pytest.raises(ac.AsyncCheckpointError):
        manager.close()
    assert visible_safetensors(save_root) == []
    assert ac.list_manifests(save_root, status=None) == []


def test_retention_keeps_groups_and_recovery_state(tmp_path):
    save_root = str(tmp_path)
    optimizer = {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": [{"lr": 1.0, "params": [0]}]}
    seq = 0
    for step in range(100, 1300, 100):
        is_recovery = step % 500 == 0
        request, _ = make_request(
            save_root, step,
            checkpoint_type=ac.CHECKPOINT_TYPE_RECOVERY if is_recovery else ac.CHECKPOINT_TYPE_PREVIEW,
            optimizer=optimizer if is_recovery else None,
        )
        seq += 1
        ac.commit_request(save_root, request, commit_seq=seq, fsync=False)

    ac.apply_retention(save_root, keep_lora_groups=3, keep_recovery_groups=1, fsync=False, log=lambda *_: None)

    groups = {m["group"]: m for m in ac.list_manifests(save_root)}
    # newest 3 groups survive, the newest of them that is a recovery (1000) keeps its optimizer
    assert sorted(groups) == ["xian_000001000", "xian_000001100", "xian_000001200"]
    assert ac.manifest_files(groups["xian_000001000"], ac.ROLE_OPTIMIZER)
    assert not os.path.exists(os.path.join(save_root, ac.CHECKPOINT_DIR, "xian_000000500"))
    assert "xian_000000500_high_noise.safetensors" not in visible_safetensors(save_root)
    assert "xian_000001100_high_noise.safetensors" in visible_safetensors(save_root)

    # trimming the recovery state keeps the LoRA preview
    ac.apply_retention(save_root, keep_lora_groups=3, keep_recovery_groups=0, fsync=False, log=lambda *_: None)
    groups = {m["group"]: m for m in ac.list_manifests(save_root)}
    assert groups["xian_000001000"]["checkpoint_type"] == ac.CHECKPOINT_TYPE_PREVIEW
    assert not ac.manifest_files(groups["xian_000001000"], ac.ROLE_OPTIMIZER)
    assert "xian_000001000_low_noise.safetensors" in visible_safetensors(save_root)


def test_recovery_lora_outlives_preview_retention(tmp_path):
    save_root = str(tmp_path)
    optimizer = {"state": {}, "param_groups": []}
    seq = 0
    for step in range(100, 900, 100):
        is_recovery = step == 500
        request, _ = make_request(
            save_root, step,
            checkpoint_type=ac.CHECKPOINT_TYPE_RECOVERY if is_recovery else ac.CHECKPOINT_TYPE_PREVIEW,
            optimizer=optimizer if is_recovery else None,
        )
        seq += 1
        ac.commit_request(save_root, request, commit_seq=seq, fsync=False)
    ac.apply_retention(save_root, keep_lora_groups=2, keep_recovery_groups=1, fsync=False, log=lambda *_: None)
    assert sorted(m["group"] for m in ac.list_manifests(save_root)) == [
        "xian_000000500", "xian_000000700", "xian_000000800",
    ]


def test_resume_never_mixes_steps(tmp_path):
    save_root = str(tmp_path)
    optimizer = {"state": {0: {"step": torch.tensor(500.0)}}, "param_groups": [{"lr": 1.0, "params": [0]}]}
    seq = 0
    for step, kind in [(500, ac.CHECKPOINT_TYPE_RECOVERY), (800, ac.CHECKPOINT_TYPE_PREVIEW)]:
        request, _ = make_request(save_root, step, checkpoint_type=kind,
                                  optimizer=optimizer if kind == ac.CHECKPOINT_TYPE_RECOVERY else None)
        seq += 1
        ac.commit_request(save_root, request, commit_seq=seq, fsync=False)

    recovery = ac.find_latest_recovery(save_root)
    assert recovery["step"] == 500
    paths = [f["path"] for f in recovery["files"]]
    assert "xian_000000500_high_noise.safetensors" in paths
    opt_path = os.path.join(save_root, ac.manifest_files(recovery, ac.ROLE_OPTIMIZER)[0]["path"])
    assert torch.load(opt_path, weights_only=True)["state"][0]["step"].item() == 500.0
    assert ac.find_latest_lora(save_root)["step"] == 800

    # a damaged recovery group is never used
    os.remove(os.path.join(save_root, "xian_000000500_low_noise.safetensors"))
    assert ac.find_latest_recovery(save_root) is None


def test_abandoned_timeline_ages_out_by_commit_order(tmp_path):
    # after resuming from 500, steps 600..800 of the old run must not outrank new saves
    save_root = str(tmp_path)
    seq = 0
    for step in [500, 600, 700, 800, 550, 600]:
        request, _ = make_request(save_root, step, seed=seq)
        seq += 1
        ac.commit_request(save_root, request, commit_seq=seq, fsync=False)
    ac.apply_retention(save_root, keep_lora_groups=2, keep_recovery_groups=1, fsync=False, log=lambda *_: None)
    assert sorted(m["step"] for m in ac.list_manifests(save_root)) == [550, 600]
