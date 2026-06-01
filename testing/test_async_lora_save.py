from collections import OrderedDict
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest


def _install_fake_dependencies(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.ModuleType("torch")

    def ones(*shape):
        return {"kind": "tensor", "shape": list(shape), "fill": 1}

    def zeros(*shape):
        return {"kind": "tensor", "shape": list(shape), "fill": 0}

    def equal(left, right):
        return left == right

    fake_torch.Tensor = dict
    fake_torch.ones = ones
    fake_torch.zeros = zeros
    fake_torch.equal = equal

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")

    def save_file(state_dict, output_path, metadata=None):
        payload = {
            "state_dict": dict(state_dict),
            "metadata": metadata,
        }
        Path(output_path).write_text(json.dumps(payload), encoding="utf-8")

    def load_file(output_path):
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
        return payload["state_dict"]

    class FakeSafeOpen:
        def __init__(self, output_path, framework="pt"):
            self._output_path = output_path

        def __enter__(self):
            payload = json.loads(Path(self._output_path).read_text(encoding="utf-8"))
            self._state_dict = payload["state_dict"]
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return self._state_dict.keys()

    fake_safetensors_torch.save_file = save_file
    fake_safetensors_torch.load_file = load_file
    fake_safetensors.safe_open = FakeSafeOpen

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    monkeypatch.delitem(sys.modules, "toolkit.async_lora_save", raising=False)


def _load_async_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_dependencies(monkeypatch)
    return importlib.import_module("toolkit.async_lora_save")


def test_write_safetensors_atomic_replaces_final_path(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    output = tmp_path / "model.safetensors"
    state_dict = OrderedDict({"lora.weight": sys.modules["torch"].ones(2, 2)})

    module.write_safetensors_atomic(state_dict, str(output), metadata={"step": "1"})

    assert output.exists()
    assert not (tmp_path / "model.safetensors.tmp").exists()
    loaded = sys.modules["safetensors.torch"].load_file(str(output))
    assert sys.modules["torch"].equal(
        loaded["lora.weight"],
        sys.modules["torch"].ones(2, 2),
    )


def test_write_safetensors_atomic_uses_os_replace(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    calls = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    output = tmp_path / "model.safetensors"

    module.write_safetensors_atomic(
        OrderedDict({"lora.weight": sys.modules["torch"].zeros(1)}),
        str(output),
        metadata=None,
    )

    assert calls == [(str(output) + ".tmp", str(output))]


def test_write_wan22_lora_snapshot_splits_high_and_low_noise(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    output = tmp_path / "demo_000000010.safetensors"
    state_dict = OrderedDict(
        {
            "diffusion_model.transformer_1.blocks.0.lora_down.weight": sys.modules[
                "torch"
            ].ones(1),
            "diffusion_model.transformer_2.blocks.0.lora_up.weight": sys.modules[
                "torch"
            ].zeros(1),
        }
    )

    written = module.write_wan22_lora_snapshot(
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
    with sys.modules["safetensors"].safe_open(str(high), framework="pt") as handle:
        assert set(handle.keys()) == {"diffusion_model.blocks.0.lora_down.weight"}
    with sys.modules["safetensors"].safe_open(str(low), framework="pt") as handle:
        assert set(handle.keys()) == {"diffusion_model.blocks.0.lora_up.weight"}
    assert not output.exists()


def test_write_wan22_lora_snapshot_uses_best_split_names(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    output = tmp_path / "demo_best.safetensors"
    state_dict = OrderedDict(
        {
            "diffusion_model.transformer_1.layer.lora_down.weight": sys.modules[
                "torch"
            ].ones(1),
            "diffusion_model.transformer_2.layer.lora_up.weight": sys.modules[
                "torch"
            ].zeros(1),
        }
    )

    module.write_wan22_lora_snapshot(
        state_dict=state_dict,
        output_path=str(output),
        metadata={},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()


def test_async_lora_save_queue_writes_and_drains(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    queue = module.AsyncLoraSaveQueue()
    output = tmp_path / "demo_best.safetensors"

    queue.submit_wan22_lora(
        state_dict=OrderedDict(
            {
                "diffusion_model.transformer_1.layer.weight": sys.modules[
                    "torch"
                ].ones(1),
                "diffusion_model.transformer_2.layer.weight": sys.modules[
                    "torch"
                ].zeros(1),
            }
        ),
        output_path=str(output),
        metadata={},
        split_multistage_loras=True,
        train_high_noise=True,
        train_low_noise=True,
    )
    queue.drain()
    queue.close()

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()


def test_async_lora_save_queue_drain_raises_worker_errors(monkeypatch, tmp_path):
    module = _load_async_module(monkeypatch)
    queue = module.AsyncLoraSaveQueue()

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
