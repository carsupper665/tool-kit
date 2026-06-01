import importlib
import sys
import types

import pytest


def _load_save_config(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})

    fake_torch_backends = types.ModuleType("torch.backends")
    fake_torch_backends.mps = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = fake_torch_backends

    fake_torchaudio = types.ModuleType("torchaudio")

    fake_torchao = types.ModuleType("torchao")
    fake_torchao_quantization = types.ModuleType("torchao.quantization")
    fake_torchao_quant_primitives = types.ModuleType(
        "torchao.quantization.quant_primitives"
    )
    fake_torchao_quant_primitives._DTYPE_TO_BIT_WIDTH = {}
    fake_torchao.quantization = fake_torchao_quantization
    fake_torchao_quantization.quant_primitives = fake_torchao_quant_primitives

    fake_album_artwork = types.ModuleType("toolkit.audio.album_artwork")
    fake_album_artwork.add_album_artwork = lambda *args, **kwargs: None

    fake_prompt_utils = types.ModuleType("toolkit.prompt_utils")
    fake_prompt_utils.PromptEmbeds = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.backends", fake_torch_backends)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_torchao_quantization)
    monkeypatch.setitem(
        sys.modules,
        "torchao.quantization.quant_primitives",
        fake_torchao_quant_primitives,
    )
    monkeypatch.setitem(sys.modules, "toolkit.audio.album_artwork", fake_album_artwork)
    monkeypatch.setitem(sys.modules, "toolkit.prompt_utils", fake_prompt_utils)
    monkeypatch.delitem(sys.modules, "toolkit.config_modules", raising=False)

    module = importlib.import_module("toolkit.config_modules")
    return module.SaveConfig


@pytest.fixture
def save_config_cls(monkeypatch: pytest.MonkeyPatch):
    try:
        yield _load_save_config(monkeypatch)
    finally:
        sys.modules.pop("toolkit.config_modules", None)


def test_save_config_defaults_keep_async_lora_queue_off(save_config_cls):
    config = save_config_cls()

    assert config.async_lora_save is False
    assert config.save_best_lora is False


def test_save_config_accepts_async_lora_queue_and_save_best(save_config_cls):
    config = save_config_cls(async_lora_save=True, save_best_lora=True)

    assert config.async_lora_save is True
    assert config.save_best_lora is True
