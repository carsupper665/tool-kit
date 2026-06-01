# Wan 2.2 LoRA Save Queue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Wan 2.2 14B only LoRA safetensors save queue and save-best path that uses raw `loss_dict["loss"]` without changing optimizer checkpoint behavior.

**Architecture:** Keep training state and optimizer saves synchronous, but move Wan 2.2 14B LoRA safetensors writes behind a tiny CPU snapshot queue. The safe boundary is `network.get_state_dict()`, which clones tensors to CPU and dtype before the queue accepts work. Best LoRA saves use the same CPU snapshot path and keep the Wan 2.2 split high and low noise filenames.

**Tech Stack:** Python, PyTorch CPU tensors, `safetensors.torch.save_file`, `threading`, `queue.Queue`, `os.replace`, pytest.

---

## Context For The Implementer

Current code references to read before editing:

- `jobs/process/BaseSDTrainProcess.py`: `save()` is synchronous. It updates metadata, calls `self.network.save_weights(...)`, saves optimizer state, calls `clean_up_saves()`, and runs `post_save_hook()`. The training loop receives `loss_dict` from `hook_train_loop()`, logs each value, and currently calls `self.save(self.step_num)` on step saves plus `self.save()` on normal finish. Metadata is updated in `update_training_metadata()` and restored in `load_training_state_from_metadata()`.
- `extensions_built_in/sd_trainer/SDTrainer.py`: `hook_train_loop()` returns an `OrderedDict` with raw `{'loss': (total_loss / len(batch_list)).item()}`. Use this raw `loss_dict["loss"]` value for save-best. Don't smooth it.
- `toolkit/network_mixins.py`: `get_state_dict()` detaches, clones, moves tensors to CPU, and casts dtype. That is the safe async boundary. `save_weights()` currently calls `get_state_dict()`, then delegates Wan saves to `base_model.save_lora(...)` when present.
- `extensions_built_in/diffusion_models/wan22/wan22_14b_model.py`: `Wan2214bModel.arch` is `"wan22_14b"`. `save_lora()` writes split files as `*_high_noise.safetensors` and `*_low_noise.safetensors` when `split_multistage_loras` is true.
- `toolkit/config_modules.py`: `SaveConfig` owns save settings. Add any new save queue and save-best fields here.

Scope rules:

- First version is only for `wan22_14b`.
- Queue handles LoRA safetensors only.
- Queue must not handle optimizer, model folder saves, adapter saves, embedding only saves, decorator saves, hub upload, samples, or UI.
- Don't queue live CUDA tensors or live model state. Only queue CPU state dict snapshots and plain metadata/config values.
- Keep existing checkpoint and optimizer semantics. Optimizer still saves synchronously as `optimizer.pt` inside `BaseSDTrainProcess.save()`.
- Best outputs for split Wan 2.2 LoRA must be `*_best_high_noise.safetensors` and `*_best_low_noise.safetensors`.
- Exclude `*_best*.safetensors` from cleanup and resume discovery.
- Drain the queue on normal finish, exception paths, and interrupt paths before the process releases training objects.

Recommended minimal module:

- Create `toolkit/async_lora_save.py` for pure CPU safetensors writing, Wan 2.2 split naming, and a small one-worker queue.
- Gate all integration from `BaseSDTrainProcess` so non Wan models keep their current synchronous path.

## Task 1: Add SaveConfig Flags

**Files:**

- Modify: `toolkit/config_modules.py:23`
- Test: `testing/test_save_config.py`

**Step 1: Write the failing tests**

Create or extend `testing/test_save_config.py`:

```python
from toolkit.config_modules import SaveConfig


def test_save_config_defaults_keep_async_lora_queue_off():
    config = SaveConfig()

    assert config.async_lora_save is False
    assert config.save_best_lora is False


def test_save_config_accepts_async_lora_queue_and_save_best():
    config = SaveConfig(async_lora_save=True, save_best_lora=True)

    assert config.async_lora_save is True
    assert config.save_best_lora is True
```

**Step 2: Run the test to verify it fails**

Run: `python -m pytest testing/test_save_config.py -q`

Expected: FAIL with an `AttributeError` for `async_lora_save` or `save_best_lora`.

**Step 3: Add the minimal config fields**

In `toolkit/config_modules.py`, add only these fields to `SaveConfig.__init__`:

```python
self.async_lora_save: bool = kwargs.get("async_lora_save", False)
self.save_best_lora: bool = kwargs.get("save_best_lora", False)
```

Keep defaults off so existing jobs behave exactly the same.

**Step 4: Run the test to verify it passes**

Run: `python -m pytest testing/test_save_config.py -q`

Expected: PASS with `2 passed`.

## Task 2: Create Atomic CPU Safetensors Writer

**Files:**

- Create: `toolkit/async_lora_save.py`
- Test: `testing/test_async_lora_save.py`

**Step 1: Write the failing atomic writer test**

Create `testing/test_async_lora_save.py`:

```python
from collections import OrderedDict
import os

import torch
from safetensors.torch import load_file

from toolkit.async_lora_save import write_safetensors_atomic


def test_write_safetensors_atomic_replaces_final_path(tmp_path):
    output = tmp_path / "model.safetensors"
    state_dict = OrderedDict({"lora.weight": torch.ones(2, 2)})

    write_safetensors_atomic(state_dict, str(output), metadata={"step": "1"})

    assert output.exists()
    assert not (tmp_path / "model.safetensors.tmp").exists()
    loaded = load_file(str(output))
    assert torch.equal(loaded["lora.weight"], torch.ones(2, 2))


def test_write_safetensors_atomic_uses_os_replace(monkeypatch, tmp_path):
    calls = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    output = tmp_path / "model.safetensors"

    write_safetensors_atomic(
        OrderedDict({"lora.weight": torch.zeros(1)}),
        str(output),
        metadata=None,
    )

    assert calls == [(str(output) + ".tmp", str(output))]
```

**Step 2: Run the test to verify it fails**

Run: `python -m pytest testing/test_async_lora_save.py::test_write_safetensors_atomic_replaces_final_path -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'toolkit.async_lora_save'`.

**Step 3: Implement the minimal writer**

Create `toolkit/async_lora_save.py` with this shape:

```python
import os
from collections import OrderedDict
from typing import Optional, Mapping

import torch
from safetensors.torch import save_file


TensorDict = Mapping[str, torch.Tensor]


def write_safetensors_atomic(
    state_dict: TensorDict,
    output_path: str,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    tmp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(OrderedDict(state_dict), tmp_path, metadata=metadata)
    os.replace(tmp_path, output_path)
```

If `save_file()` raises, leave no final partial file. If a temp file remains after an exception, later tasks may remove it before retrying, but don't hide the original exception.

**Step 4: Run the atomic writer tests**

Run: `python -m pytest testing/test_async_lora_save.py -q`

Expected: PASS for the two tests in this task.

## Task 3: Add Wan 2.2 Split LoRA Snapshot Writer

**Files:**

- Modify: `toolkit/async_lora_save.py`
- Test: `testing/test_async_lora_save.py`

**Step 1: Write the failing split tests**

Append these tests:

```python
from safetensors.torch import safe_open

from toolkit.async_lora_save import write_wan22_lora_snapshot


def _keys(path):
    with safe_open(str(path), framework="pt") as handle:
        return set(handle.keys())


def test_write_wan22_lora_snapshot_splits_high_and_low_noise(tmp_path):
    output = tmp_path / "demo_000000010.safetensors"
    state_dict = OrderedDict({
        "diffusion_model.transformer_1.blocks.0.lora_down.weight": torch.ones(1),
        "diffusion_model.transformer_2.blocks.0.lora_up.weight": torch.zeros(1),
    })

    written = write_wan22_lora_snapshot(
        state_dict=state_dict,
        output_path=str(output),
        metadata={"step": "10"},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )

    high = tmp_path / "demo_000000010_high_noise.safetensors"
    low = tmp_path / "demo_000000010_low_noise.safetensors"
    assert written == [str(high), str(low)]
    assert high.exists()
    assert low.exists()
    assert _keys(high) == {"diffusion_model.blocks.0.lora_down.weight"}
    assert _keys(low) == {"diffusion_model.blocks.0.lora_up.weight"}
    assert not output.exists()


def test_write_wan22_lora_snapshot_uses_best_split_names(tmp_path):
    output = tmp_path / "demo_best.safetensors"
    state_dict = OrderedDict({
        "diffusion_model.transformer_1.layer.lora_down.weight": torch.ones(1),
        "diffusion_model.transformer_2.layer.lora_up.weight": torch.zeros(1),
    })

    write_wan22_lora_snapshot(
        state_dict=state_dict,
        output_path=str(output),
        metadata={},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()
```

**Step 2: Run the split tests to verify they fail**

Run: `python -m pytest testing/test_async_lora_save.py::test_write_wan22_lora_snapshot_splits_high_and_low_noise testing/test_async_lora_save.py::test_write_wan22_lora_snapshot_uses_best_split_names -q`

Expected: FAIL with `ImportError` for `write_wan22_lora_snapshot`.

**Step 3: Implement the Wan 2.2 snapshot writer**

Add this function to `toolkit/async_lora_save.py`:

```python
def write_wan22_lora_snapshot(
    state_dict: TensorDict,
    output_path: str,
    metadata: Optional[dict[str, str]],
    split_multistage_loras: bool,
    train_high_noise: bool,
    train_low_noise: bool,
) -> list[str]:
    if not split_multistage_loras:
        write_safetensors_atomic(state_dict, output_path, metadata=metadata)
        return [output_path]

    high_noise_lora = OrderedDict()
    low_noise_lora = OrderedDict()
    only_train_high_noise = train_high_noise and not train_low_noise
    only_train_low_noise = train_low_noise and not train_high_noise

    for key, value in state_dict.items():
        if ".transformer_1." in key or only_train_high_noise:
            high_noise_lora[key.replace(".transformer_1.", ".")] = value
        elif ".transformer_2." in key or only_train_low_noise:
            low_noise_lora[key.replace(".transformer_2.", ".")] = value

    written = []
    if high_noise_lora:
        high_path = output_path.replace(".safetensors", "_high_noise.safetensors")
        write_safetensors_atomic(high_noise_lora, high_path, metadata=metadata)
        written.append(high_path)
    if low_noise_lora:
        low_path = output_path.replace(".safetensors", "_low_noise.safetensors")
        write_safetensors_atomic(low_noise_lora, low_path, metadata=metadata)
        written.append(low_path)
    return written
```

This mirrors `Wan2214bModel.save_lora()` without reading live model tensors in the worker.

**Step 4: Run the helper tests**

Run: `python -m pytest testing/test_async_lora_save.py -q`

Expected: PASS for all helper tests.

## Task 4: Add A One Worker LoRA Save Queue

**Files:**

- Modify: `toolkit/async_lora_save.py`
- Test: `testing/test_async_lora_save.py`

**Step 1: Write the failing queue tests**

Append these tests:

```python
import pytest

from toolkit.async_lora_save import AsyncLoraSaveQueue


def test_async_lora_save_queue_writes_and_drains(tmp_path):
    queue = AsyncLoraSaveQueue()
    output = tmp_path / "demo_best.safetensors"

    queue.submit_wan22_lora(
        state_dict=OrderedDict({
            "diffusion_model.transformer_1.layer.weight": torch.ones(1),
            "diffusion_model.transformer_2.layer.weight": torch.zeros(1),
        }),
        output_path=str(output),
        metadata={},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )
    queue.drain()

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()


def test_async_lora_save_queue_drain_raises_worker_errors(tmp_path):
    queue = AsyncLoraSaveQueue()
    queue.submit_wan22_lora(
        state_dict={"not_a_tensor": object()},
        output_path=str(tmp_path / "broken.safetensors"),
        metadata={},
        split_multistage_loras=False,
        train_high_noise=True,
        train_low_noise=True,
    )

    with pytest.raises(Exception):
        queue.drain()
```

**Step 2: Run the queue tests to verify they fail**

Run: `python -m pytest testing/test_async_lora_save.py::test_async_lora_save_queue_writes_and_drains testing/test_async_lora_save.py::test_async_lora_save_queue_drain_raises_worker_errors -q`

Expected: FAIL with `ImportError` for `AsyncLoraSaveQueue`.

**Step 3: Implement the minimal queue**

Add a one-worker queue to `toolkit/async_lora_save.py`:

```python
import queue
import threading
from dataclasses import dataclass


@dataclass
class Wan22LoraSaveJob:
    state_dict: TensorDict
    output_path: str
    metadata: Optional[dict[str, str]]
    split_multistage_loras: bool
    train_high_noise: bool
    train_low_noise: bool


class AsyncLoraSaveQueue:
    def __init__(self) -> None:
        self._jobs: queue.Queue[Wan22LoraSaveJob | None] = queue.Queue()
        self._errors: list[BaseException] = []
        self._worker = threading.Thread(target=self._run, name="aitk-lora-save", daemon=False)
        self._worker.start()

    def submit_wan22_lora(self, **kwargs) -> None:
        self._raise_if_errors()
        self._jobs.put(Wan22LoraSaveJob(**kwargs))

    def drain(self) -> None:
        self._jobs.join()
        self._raise_if_errors()

    def close(self) -> None:
        self.drain()
        self._jobs.put(None)
        self._worker.join()
        self._raise_if_errors()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                write_wan22_lora_snapshot(**job.__dict__)
            except BaseException as error:
                self._errors.append(error)
            finally:
                self._jobs.task_done()

    def _raise_if_errors(self) -> None:
        if self._errors:
            raise self._errors[0]
```

Keep it one worker. More workers can reorder saves and complicate failures. Don't add retries.

**Step 4: Run the queue tests**

Run: `python -m pytest testing/test_async_lora_save.py -q`

Expected: PASS for all helper and queue tests.

## Task 5: Gate Async Queue Setup To Wan 2.2 14B LoRA Saves

**Files:**

- Modify: `jobs/process/BaseSDTrainProcess.py:77`
- Test: `testing/test_wan22_lora_async_save.py`

**Step 1: Write the failing gate tests**

Create `testing/test_wan22_lora_async_save.py` with small fakes. The exact imports may need local test path adjustment, but keep the fake objects CPU only:

```python
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def _process_with(arch="wan22_14b", async_lora_save=True, network=True):
    process = object.__new__(BaseSDTrainProcess)
    process.save_config = SimpleNamespace(async_lora_save=async_lora_save)
    process.network = object() if network else None
    process.train_config = SimpleNamespace(merge_network_on_save=False)
    process.save_config.save_format = "safetensors"
    process.sd = SimpleNamespace(arch=arch)
    return process


def test_async_lora_save_gate_allows_only_wan22_14b_lora_safetensors():
    process = _process_with()

    assert process._should_use_async_wan22_lora_save() is True


def test_async_lora_save_gate_rejects_non_wan_models():
    process = _process_with(arch="flux")

    assert process._should_use_async_wan22_lora_save() is False


def test_async_lora_save_gate_rejects_when_config_off():
    process = _process_with(async_lora_save=False)

    assert process._should_use_async_wan22_lora_save() is False


def test_async_lora_save_gate_rejects_without_network():
    process = _process_with(network=False)

    assert process._should_use_async_wan22_lora_save() is False
```

**Step 2: Run the gate tests to verify they fail**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: FAIL with `AttributeError: 'BaseSDTrainProcess' object has no attribute '_should_use_async_wan22_lora_save'`.

**Step 3: Add the gate and lazy queue fields**

In `BaseSDTrainProcess.__init__`, add:

```python
self._async_lora_save_queue = None
self._best_lora_loss = None
```

Add methods near `save()`:

```python
def _should_use_async_wan22_lora_save(self):
    return (
        getattr(self.save_config, "async_lora_save", False)
        and self.network is not None
        and not self.train_config.merge_network_on_save
        and self.save_config.save_format == "safetensors"
        and getattr(self.sd, "arch", None) == "wan22_14b"
    )

def _get_async_lora_save_queue(self):
    if self._async_lora_save_queue is None:
        from toolkit.async_lora_save import AsyncLoraSaveQueue
        self._async_lora_save_queue = AsyncLoraSaveQueue()
    return self._async_lora_save_queue

def _drain_async_lora_save_queue(self):
    if self._async_lora_save_queue is not None:
        self._async_lora_save_queue.drain()

def _close_async_lora_save_queue(self):
    if self._async_lora_save_queue is not None:
        self._async_lora_save_queue.close()
        self._async_lora_save_queue = None
```

**Step 4: Run the gate tests**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: PASS for the gate tests.

## Task 6: Queue LoRA Snapshots From save() Without Queuing Optimizer

**Files:**

- Modify: `jobs/process/BaseSDTrainProcess.py:490`
- Test: `testing/test_wan22_lora_async_save.py`

**Step 1: Write the failing snapshot test**

Append this test using fakes. It checks behavior, not Wan weights:

```python
from collections import OrderedDict
import torch


class FakeNetwork:
    def __init__(self):
        self.multiplier = 0.5
        self.network_config = SimpleNamespace(split_multistage_loras=True)
        self.saved_sync = False

    def get_state_dict(self, extra_state_dict=None, dtype=torch.float16):
        assert self.multiplier == 1.0
        return OrderedDict({
            "diffusion_model.transformer_1.layer.weight": torch.ones(1, dtype=dtype),
            "diffusion_model.transformer_2.layer.weight": torch.zeros(1, dtype=dtype),
        })

    def save_weights(self, *args, **kwargs):
        self.saved_sync = True


class RecordingQueue:
    def __init__(self):
        self.jobs = []

    def submit_wan22_lora(self, **kwargs):
        self.jobs.append(kwargs)


def test_save_queues_cpu_lora_snapshot_and_keeps_optimizer_sync(tmp_path):
    process = _process_with()
    process.save_root = str(tmp_path)
    process.job = SimpleNamespace(name="demo")
    process.name = "demo"
    process.named_lora = False
    process.network = FakeNetwork()
    process.embedding = None
    process.embed_config = None
    process.decorator = None
    process.adapter = None
    process.adapter_config = None
    process.is_fine_tuning = False
    process.meta = {}
    process.optimizer = None
    process.ema = None
    process.snr_gos = None
    process.step_num = 10
    process.epoch_num = 0
    process.accelerator = SimpleNamespace(is_main_process=True, is_local_main_process=True)
    process.clean_up_saves = lambda: None
    process.post_save_hook = lambda path: None
    queue = RecordingQueue()
    process._get_async_lora_save_queue = lambda: queue

    process.save(step=10)

    assert process.network.saved_sync is False
    assert process.network.multiplier == 0.5
    assert queue.jobs[0]["output_path"].endswith("demo_000000010.safetensors")
    assert queue.jobs[0]["split_multistage_loras"] is True
    assert queue.jobs[0]["train_high_noise"] is True
    assert queue.jobs[0]["train_low_noise"] is True
```

**Step 2: Run the snapshot test to verify it fails**

Run: `python -m pytest testing/test_wan22_lora_async_save.py::test_save_queues_cpu_lora_snapshot_and_keeps_optimizer_sync -q`

Expected: FAIL because `save()` still calls `network.save_weights()`.

**Step 3: Change only the Wan 2.2 LoRA branch in save()**

Inside `BaseSDTrainProcess.save()`, keep the existing code path for all non gated cases. In the `if self.network is not None:` block, replace only the `self.network.save_weights(...)` call with a branch:

```python
if self._should_use_async_wan22_lora_save():
    save_dict = self.network.get_state_dict(
        extra_state_dict=embedding_dict,
        dtype=get_torch_dtype(self.save_config.dtype),
    )
    self._get_async_lora_save_queue().submit_wan22_lora(
        state_dict=save_dict,
        output_path=file_path,
        metadata=save_meta,
        split_multistage_loras=self.network.network_config.split_multistage_loras,
        train_high_noise=getattr(self.sd, "train_high_noise", True),
        train_low_noise=getattr(self.sd, "train_low_noise", True),
    )
else:
    self.network.save_weights(
        file_path,
        dtype=get_torch_dtype(self.save_config.dtype),
        metadata=save_meta,
        extra_state_dict=embedding_dict,
    )
```

Keep the existing `prev_multiplier` restore exactly as it is. Keep optimizer save code unchanged and outside the async branch.

**Step 4: Run the integration test**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: PASS for gate and snapshot tests.

## Task 7: Track Raw Loss And Save Best LoRA Snapshots

**Files:**

- Modify: `jobs/process/BaseSDTrainProcess.py:2221`
- Test: `testing/test_wan22_lora_async_save.py`

**Step 1: Write the failing save-best tests**

Append focused tests for raw loss comparison:

```python
def test_save_best_uses_raw_loss_and_queues_best_name(tmp_path):
    process = _process_with()
    process.save_config.save_best_lora = True
    process.save_root = str(tmp_path)
    process.job = SimpleNamespace(name="demo")
    process.name = "demo"
    process.network = FakeNetwork()
    process.embedding = None
    process.meta = {}
    process.step_num = 11
    process.epoch_num = 0
    process.sd.train_high_noise = True
    process.sd.train_low_noise = True
    queue = RecordingQueue()
    process._get_async_lora_save_queue = lambda: queue
    process._best_lora_loss = None

    assert process._maybe_queue_best_wan22_lora({"loss": 0.42}) is True
    assert process._best_lora_loss == 0.42
    assert queue.jobs[0]["output_path"].endswith("demo_best.safetensors")

    assert process._maybe_queue_best_wan22_lora({"loss": 0.50}) is False
    assert len(queue.jobs) == 1

    assert process._maybe_queue_best_wan22_lora({"loss": 0.41}) is True
    assert process._best_lora_loss == 0.41
    assert len(queue.jobs) == 2


def test_save_best_ignores_missing_loss_or_config_off(tmp_path):
    process = _process_with()
    process.save_config.save_best_lora = False

    assert process._maybe_queue_best_wan22_lora({"loss": 0.1}) is False

    process.save_config.save_best_lora = True
    assert process._maybe_queue_best_wan22_lora({}) is False
```

**Step 2: Run the save-best tests to verify they fail**

Run: `python -m pytest testing/test_wan22_lora_async_save.py::test_save_best_uses_raw_loss_and_queues_best_name testing/test_wan22_lora_async_save.py::test_save_best_ignores_missing_loss_or_config_off -q`

Expected: FAIL with `AttributeError` for `_maybe_queue_best_wan22_lora`.

**Step 3: Add a shared snapshot queue helper**

To avoid duplicating snapshot code, extract the queue submit logic into a helper:

```python
def _queue_wan22_lora_snapshot(self, file_path, save_meta):
    embedding_dict = self.embedding.state_dict() if self.embedding else None
    prev_multiplier = self.network.multiplier
    self.network.multiplier = 1.0
    try:
        save_dict = self.network.get_state_dict(
            extra_state_dict=embedding_dict,
            dtype=get_torch_dtype(self.save_config.dtype),
        )
    finally:
        self.network.multiplier = prev_multiplier

    self._get_async_lora_save_queue().submit_wan22_lora(
        state_dict=save_dict,
        output_path=file_path,
        metadata=save_meta,
        split_multistage_loras=self.network.network_config.split_multistage_loras,
        train_high_noise=getattr(self.sd, "train_high_noise", True),
        train_low_noise=getattr(self.sd, "train_low_noise", True),
    )
```

Use this helper from `save()` only inside the Wan gated branch.

**Step 4: Add the save-best method**

Add:

```python
def _maybe_queue_best_wan22_lora(self, loss_dict):
    if not getattr(self.save_config, "save_best_lora", False):
        return False
    if not self._should_use_async_wan22_lora_save():
        return False
    if loss_dict is None or "loss" not in loss_dict:
        return False

    raw_loss = float(loss_dict["loss"])
    if self._best_lora_loss is not None and raw_loss >= self._best_lora_loss:
        return False

    self._best_lora_loss = raw_loss
    self.update_training_metadata()
    save_meta = get_meta_for_safetensors(copy.deepcopy(self.meta), self.job.name)
    output_path = os.path.join(self.save_root, f"{self.job.name}_best.safetensors")
    self._queue_wan22_lora_snapshot(output_path, save_meta)
    return True
```

This uses raw `loss_dict["loss"]` from `SDTrainer.hook_train_loop()`. Don't average it again. Don't compare logger values.

**Step 5: Call save-best in the training loop after a successful train step**

In `BaseSDTrainProcess.run()`, after `learning_rate` and progress bar updates are computed and only when `not did_oom and loss_dict is not None`, call:

```python
self._maybe_queue_best_wan22_lora(loss_dict)
```

Place this before step save and sample work. Don't call it on OOM, missing loss, or before the first successful loss.

**Step 6: Run the save-best tests**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: PASS for all Wan async integration tests.

## Task 8: Exclude Best LoRAs From Cleanup And Resume

**Files:**

- Modify: `jobs/process/BaseSDTrainProcess.py:403`
- Modify: `jobs/process/BaseSDTrainProcess.py:797`
- Test: `testing/test_wan22_lora_async_save.py`

**Step 1: Write the failing cleanup and resume tests**

Append tests that make tiny files, no model needed:

```python
def test_cleanup_does_not_remove_best_lora_files(tmp_path):
    process = _process_with()
    process.save_root = str(tmp_path)
    process.job = SimpleNamespace(name="demo")
    process.embed_config = None
    process.sd.max_step_saves_to_keep_multiplier = 2
    process.save_config.max_step_saves_to_keep = 1
    process.accelerator = SimpleNamespace(is_main_process=True)

    for name in [
        "demo_000000001_high_noise.safetensors",
        "demo_000000001_low_noise.safetensors",
        "demo_best_high_noise.safetensors",
        "demo_best_low_noise.safetensors",
    ]:
        (tmp_path / name).write_text("x")

    process.clean_up_saves()

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()


def test_get_latest_save_path_ignores_best_lora_files(tmp_path):
    process = _process_with()
    process.save_root = str(tmp_path)
    process.job = SimpleNamespace(name="demo")
    process.network_config = SimpleNamespace(pretrained_lora_path=None)
    (tmp_path / "demo_best_high_noise.safetensors").write_text("x")

    assert process.get_latest_save_path("demo") is None
```

**Step 2: Run the tests to verify they fail**

Run: `python -m pytest testing/test_wan22_lora_async_save.py::test_cleanup_does_not_remove_best_lora_files testing/test_wan22_lora_async_save.py::test_get_latest_save_path_ignores_best_lora_files -q`

Expected: FAIL because best files are included in cleanup or resume glob matches.

**Step 3: Exclude best files in cleanup**

In `clean_up_saves()`, after collecting `safetensors_files`, filter them:

```python
safetensors_files = [f for f in safetensors_files if "_best" not in os.path.basename(f)]
```

Do this before sorting and removal list creation. Keep `max_step_saves_to_keep_multiplier` behavior for normal high and low step saves.

**Step 4: Exclude best files in resume discovery**

In `get_latest_save_path()`, after `paths = [p for p in paths if os.path.exists(p)]`, add:

```python
paths = [p for p in paths if "_best" not in os.path.basename(p)]
```

This prevents best files from becoming resume checkpoints. The latest normal step save remains the resume source.

**Step 5: Run the tests**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: PASS for all Wan async integration tests.

## Task 9: Drain The Queue On Finish, Exception, And Interrupt

**Files:**

- Modify: `jobs/process/BaseSDTrainProcess.py:2130`
- Test: `testing/test_wan22_lora_async_save.py`

**Step 1: Write failing drain tests**

Add small method tests around the close helper and train loop exception shape:

```python
class CloseRecordingQueue:
    def __init__(self):
        self.closed = False
        self.drained = False

    def drain(self):
        self.drained = True

    def close(self):
        self.closed = True


def test_close_async_lora_save_queue_closes_and_clears_queue():
    process = _process_with()
    queue = CloseRecordingQueue()
    process._async_lora_save_queue = queue

    process._close_async_lora_save_queue()

    assert queue.closed is True
    assert process._async_lora_save_queue is None
```

**Step 2: Run the drain test to verify it fails if close is incomplete**

Run: `python -m pytest testing/test_wan22_lora_async_save.py::test_close_async_lora_save_queue_closes_and_clears_queue -q`

Expected: PASS if Task 5 added `_close_async_lora_save_queue()` correctly. If it fails, fix that helper before touching the train loop.

**Step 3: Wrap the training body with queue close**

In `BaseSDTrainProcess.run()`, use `try` and `finally` around the train loop work after setup has created config fields. The goal is:

```python
try:
    ... existing run body ...
finally:
    self._close_async_lora_save_queue()
```

Place the finalizer so it runs for:

- Normal finish after final `self.save()`.
- `KeyboardInterrupt`.
- Exceptions from training, sampling, saving, logging, or hub upload.

Don't swallow exceptions. The finalizer must close the queue and then re-raise the original failure unless the queue close itself raises first. If the queue raises because a worker save failed, let that error surface.

**Step 4: Drain before final cleanup of model objects**

Before `del (self.sd, unet, noise_scheduler, optimizer, self.network, tokenizer, text_encoder)`, call:

```python
self._close_async_lora_save_queue()
```

If the full `try/finally` from Step 3 already covers this, make the helper idempotent so a second call is safe. This ensures queued CPU writes finish before training objects are released.

**Step 5: Run drain-related tests**

Run: `python -m pytest testing/test_wan22_lora_async_save.py -q`

Expected: PASS for all Wan async integration tests.

## Task 10: Compile And Run Focused Tests

**Files:**

- Verify: `toolkit/async_lora_save.py`
- Verify: `toolkit/config_modules.py`
- Verify: `jobs/process/BaseSDTrainProcess.py`
- Verify: `testing/test_async_lora_save.py`
- Verify: `testing/test_save_config.py`
- Verify: `testing/test_wan22_lora_async_save.py`

**Step 1: Run Python compile checks**

Run: `python -m py_compile toolkit/async_lora_save.py toolkit/config_modules.py jobs/process/BaseSDTrainProcess.py testing/test_async_lora_save.py testing/test_save_config.py testing/test_wan22_lora_async_save.py`

Expected: exit code 0 and no output.

**Step 2: Run focused unit tests**

Run: `python -m pytest testing/test_async_lora_save.py testing/test_save_config.py testing/test_wan22_lora_async_save.py -q`

Expected: PASS for all focused tests.

**Step 3: Run the nearest existing save tests if present**

First inspect tests under `testing/` for existing save or training process tests. If a matching file exists, run it. Example command if present:

Run: `python -m pytest testing -q`

Expected: PASS, or only pre-existing unrelated failures documented with exact failing test names.

**Step 4: Manual verification using fake CPU tensors**

Run this one-off driver from repo root:

```bash
python - <<'PY'
from collections import OrderedDict
import tempfile
from pathlib import Path
import torch
from safetensors.torch import safe_open
from toolkit.async_lora_save import AsyncLoraSaveQueue

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "demo_best.safetensors"
    queue = AsyncLoraSaveQueue()
    queue.submit_wan22_lora(
        state_dict=OrderedDict({
            "diffusion_model.transformer_1.layer.weight": torch.ones(1),
            "diffusion_model.transformer_2.layer.weight": torch.zeros(1),
        }),
        output_path=str(output),
        metadata={"step": "7"},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )
    queue.close()
    high = output.with_name("demo_best_high_noise.safetensors")
    low = output.with_name("demo_best_low_noise.safetensors")
    assert high.exists(), high
    assert low.exists(), low
    with safe_open(str(high), framework="pt") as handle:
        assert "diffusion_model.layer.weight" in handle.keys()
    with safe_open(str(low), framework="pt") as handle:
        assert "diffusion_model.layer.weight" in handle.keys()
print("wan22 async lora save queue manual check passed")
PY
```

Expected: prints `wan22 async lora save queue manual check passed` and exits 0.

## Final Acceptance Criteria

- `SaveConfig` has `async_lora_save` and `save_best_lora`, both defaulting to `False`.
- Async save queue activates only when all of these are true: `save.async_lora_save`, `sd.arch == "wan22_14b"`, LoRA network exists, `save_format == "safetensors"`, and `merge_network_on_save` is false.
- Queue jobs contain only CPU tensor snapshots from `network.get_state_dict()`, metadata, output path, and plain split flags.
- Normal LoRA step saves still produce Wan 2.2 high and low split files when `split_multistage_loras` is true.
- Save-best compares raw `loss_dict["loss"]` and writes `*_best_high_noise.safetensors` and `*_best_low_noise.safetensors` for split Wan 2.2 14B LoRA.
- `*_best*.safetensors` files are ignored by cleanup and resume discovery.
- Optimizer save stays synchronous and remains `optimizer.pt`.
- Queue drains or closes on normal finish, exceptions, and interrupts.
- Atomic writes use temp files and `os.replace` for final placement.
- Focused tests, compile checks, and the fake CPU tensor manual check pass.

## Out Of Scope

- Non Wan models.
- Wan 2.1, Wan 2.2 5B, Flux, SDXL, UI, and hub upload behavior.
- Optimizer async saves.
- Background saving from live CUDA tensors or live model state.
- Changing checkpoint retention rules for normal step checkpoints beyond excluding best files.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-06-01-wan22-lora-save-queue.md`. Two execution options:

1. **Subagent-Driven (this session)** - Dispatch a fresh subagent per task, review between tasks, and iterate quickly in this session. Required sub-skill: `superpowers:subagent-driven-development`.
2. **Parallel Session (separate)** - Open a new session in the dedicated worktree and implement with `superpowers:executing-plans`, batching execution with checkpoints.
