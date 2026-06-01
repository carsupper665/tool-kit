import importlib.util
from collections import OrderedDict
from pathlib import Path
import sys
import types

import pytest


def _make_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_base_process_dependencies(monkeypatch: pytest.MonkeyPatch):
    fake_torch = _make_stub_module(
        "torch",
        Tensor=dict,
        float16="float16",
        float32="float32",
        save=lambda *args, **kwargs: None,
        device=lambda *args, **kwargs: "cpu",
    )
    fake_torch.nn = types.SimpleNamespace(Module=object, Parameter=object)
    fake_torch.optim = types.SimpleNamespace(Optimizer=object)
    fake_torch_backends = _make_stub_module("torch.backends")
    fake_torch_backends.__path__ = []
    fake_torch_backends_cuda = _make_stub_module("torch.backends.cuda")
    fake_torch_backends.cuda = fake_torch_backends_cuda
    fake_torch.backends = fake_torch_backends
    fake_torch.utils = _make_stub_module("torch.utils")
    fake_torch_utils_data = _make_stub_module("torch.utils.data", DataLoader=object)
    fake_torch.profiler = types.SimpleNamespace(
        profile=lambda *args, **kwargs: None,
        ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )

    fake_numpy = _make_stub_module("numpy")
    fake_yaml = _make_stub_module(
        "yaml",
        dump=lambda *args, **kwargs: None,
        load=lambda *args, **kwargs: {},
        FullLoader=object,
    )

    fake_safetensors = _make_stub_module("safetensors")
    fake_safetensors_torch = _make_stub_module(
        "safetensors.torch",
        save_file=lambda *args, **kwargs: None,
        load_file=lambda *args, **kwargs: {},
    )

    fake_diffusers = _make_stub_module(
        "diffusers",
        T2IAdapter=object,
        ControlNetModel=object,
        FluxTransformer2DModel=object,
    )
    fake_diffusers_training_utils = _make_stub_module(
        "diffusers.training_utils",
        compute_density_for_timestep_sampling=lambda *args, **kwargs: None,
    )

    fake_hf_hub = _make_stub_module(
        "huggingface_hub",
        HfApi=object,
        interpreter_login=lambda *args, **kwargs: None,
    )

    fake_transformers = _make_stub_module(
        "transformers",
        utils=types.SimpleNamespace(
            logging=types.SimpleNamespace(
                set_verbosity_warning=lambda: None,
                set_verbosity_error=lambda: None,
            )
        ),
    )
    fake_accelerate = _make_stub_module("accelerate", Accelerator=object)

    fake_toolkit_accelerator = _make_stub_module(
        "toolkit.accelerator",
        get_accelerator=lambda: types.SimpleNamespace(
            is_local_main_process=True,
            is_main_process=True,
            device="cpu",
        ),
        unwrap_model=lambda x: x,
    )
    fake_toolkit_logging = _make_stub_module(
        "toolkit.logging_aitk",
        create_logger=lambda *args, **kwargs: types.SimpleNamespace(
            log=lambda *a, **k: None,
            commit=lambda *a, **k: None,
            finish=lambda *a, **k: None,
        ),
    )
    fake_toolkit_metadata = _make_stub_module(
        "toolkit.metadata",
        get_meta_for_safetensors=lambda meta, name=None, add_software_info=True: meta,
        load_metadata_from_safetensors=lambda path: {},
        add_base_model_info_to_meta=lambda *args, **kwargs: None,
        parse_metadata_from_safetensors=lambda meta: meta,
    )
    fake_toolkit_train_tools = _make_stub_module(
        "toolkit.train_tools",
        get_torch_dtype=lambda value: value,
        LearnableSNRGamma=object,
        apply_learnable_snr_gos=lambda *args, **kwargs: None,
        apply_snr_weight=lambda *args, **kwargs: None,
    )
    fake_toolkit_config_modules = _make_stub_module(
        "toolkit.config_modules",
        SaveConfig=object,
        LoggingConfig=object,
        SampleConfig=object,
        NetworkConfig=object,
        TrainConfig=object,
        ModelConfig=object,
        GenerateImageConfig=object,
        EmbeddingConfig=object,
        DatasetConfig=object,
        preprocess_dataset_raw_config=lambda raw: raw,
        AdapterConfig=object,
        GuidanceConfig=object,
        validate_configs=lambda *args, **kwargs: None,
        DecoratorConfig=object,
    )

    simple_toolkit_modules = {
        "toolkit.memory_management": {"MemoryManager": object},
        "toolkit.basic": {
            "value_map": lambda *args, **kwargs: None,
            "flush": lambda: None,
        },
        "toolkit.clip_vision_adapter": {"ClipVisionAdapter": object},
        "toolkit.custom_adapter": {"CustomAdapter": object},
        "toolkit.data_loader": {
            "get_dataloader_from_datasets": lambda *args, **kwargs: None,
            "trigger_dataloader_setup_epoch": lambda *args, **kwargs: None,
        },
        "toolkit.data_transfer_object.data_loader": {
            "FileItemDTO": object,
            "DataLoaderBatchDTO": object,
        },
        "toolkit.ema": {"ExponentialMovingAverage": object},
        "toolkit.embedding": {"Embedding": object},
        "toolkit.image_utils": {
            "show_tensors": lambda *args, **kwargs: None,
            "show_latents": lambda *args, **kwargs: None,
            "reduce_contrast": lambda *args, **kwargs: None,
        },
        "toolkit.ip_adapter": {"IPAdapter": object},
        "toolkit.lora_special": {"LoRASpecialNetwork": object},
        "toolkit.lorm": {
            "convert_diffusers_unet_to_lorm": lambda *args, **kwargs: None,
            "count_parameters": lambda *args, **kwargs: 0,
            "print_lorm_extract_details": lambda *args, **kwargs: None,
            "lorm_ignore_if_contains": lambda *args, **kwargs: None,
            "lorm_parameter_threshold": lambda *args, **kwargs: None,
            "LORM_TARGET_REPLACE_MODULE": object,
        },
        "toolkit.lycoris_special": {"LycorisSpecialNetwork": object},
        "toolkit.models.decorator": {"Decorator": object},
        "toolkit.network_mixins": {"Network": object},
        "toolkit.optimizer": {"get_optimizer": lambda *args, **kwargs: None},
        "toolkit.paths": {"CONFIG_ROOT": "config"},
        "toolkit.progress_bar": {"ToolkitProgressBar": object},
        "toolkit.reference_adapter": {"ReferenceAdapter": object},
        "toolkit.sampler": {"get_sampler": lambda *args, **kwargs: None},
        "toolkit.saving": {
            "save_t2i_from_diffusers": lambda *args, **kwargs: None,
            "load_t2i_model": lambda *args, **kwargs: None,
            "save_ip_adapter_from_diffusers": lambda *args, **kwargs: None,
            "load_ip_adapter_model": lambda *args, **kwargs: None,
            "load_custom_adapter_model": lambda *args, **kwargs: None,
        },
        "toolkit.scheduler": {"get_lr_scheduler": lambda *args, **kwargs: None},
        "toolkit.sd_device_states_presets": {
            "get_train_sd_device_state_preset": lambda *args, **kwargs: None,
        },
        "toolkit.stable_diffusion_model": {"StableDiffusion": object},
        "toolkit.util.blended_blur_noise": {
            "get_blended_blur_noise": lambda *args, **kwargs: None,
        },
        "toolkit.util.get_model": {"get_model_class": lambda *args, **kwargs: None},
        "toolkit.print": {"print_acc": lambda *args, **kwargs: None},
    }

    fake_jobs_process = _make_stub_module("jobs.process")
    fake_jobs_process.BaseTrainProcess = type("BaseTrainProcess", (), {})
    fake_jobs_process.__path__ = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.backends", fake_torch.backends)
    monkeypatch.setitem(sys.modules, "torch.backends.cuda", fake_torch_backends_cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.training_utils", fake_diffusers_training_utils)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "toolkit.accelerator", fake_toolkit_accelerator)
    monkeypatch.setitem(sys.modules, "toolkit.logging_aitk", fake_toolkit_logging)
    monkeypatch.setitem(sys.modules, "toolkit.metadata", fake_toolkit_metadata)
    monkeypatch.setitem(sys.modules, "toolkit.train_tools", fake_toolkit_train_tools)
    monkeypatch.setitem(sys.modules, "toolkit.config_modules", fake_toolkit_config_modules)
    monkeypatch.setitem(sys.modules, "jobs.process", fake_jobs_process)

    for module_name, attrs in simple_toolkit_modules.items():
        monkeypatch.setitem(sys.modules, module_name, _make_stub_module(module_name, **attrs))

    monkeypatch.delitem(sys.modules, "jobs.process.BaseSDTrainProcess", raising=False)


def _load_base_process_module(monkeypatch: pytest.MonkeyPatch):
    _install_base_process_dependencies(monkeypatch)
    module_name = "jobs.process.BaseSDTrainProcess"
    module_path = Path(__file__).resolve().parents[1] / "jobs" / "process" / "BaseSDTrainProcess.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _process_with(base_process_cls, arch="wan22_14b", async_lora_save=True, network=True):
    process = object.__new__(base_process_cls)
    process.save_config = types.SimpleNamespace(
        async_lora_save=async_lora_save,
        save_format="safetensors",
        save_best_lora=False,
        dtype="float16",
        max_step_saves_to_keep=1,
    )
    process.network = object() if network else None
    process.train_config = types.SimpleNamespace(merge_network_on_save=False)
    process.sd = types.SimpleNamespace(arch=arch)
    return process


class FakeNetwork:
    def __init__(self):
        self.multiplier = 0.5
        self.network_config = types.SimpleNamespace(split_multistage_loras=True)
        self.saved_sync = False
        self.calls = []

    def get_state_dict(self, extra_state_dict=None, dtype=None):
        self.calls.append((extra_state_dict, dtype, self.multiplier))
        assert self.multiplier == 1.0
        return OrderedDict(
            {
                "diffusion_model.transformer_1.layer.weight": {"fill": 1, "dtype": dtype},
                "diffusion_model.transformer_2.layer.weight": {"fill": 0, "dtype": dtype},
            }
        )

    def save_weights(self, *args, **kwargs):
        self.saved_sync = True


class RecordingQueue:
    def __init__(self):
        self.jobs = []

    def submit_wan22_lora(self, **kwargs):
        self.jobs.append(kwargs)


class CloseRecordingQueue:
    def __init__(self):
        self.closed = False
        self.drained = False

    def drain(self):
        self.drained = True

    def close(self):
        self.closed = True


def test_async_lora_save_gate_allows_only_wan22_14b_lora_safetensors(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)

    assert process._should_use_async_wan22_lora_save() is True


def test_async_lora_save_gate_rejects_non_wan_models(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess, arch="flux")

    assert process._should_use_async_wan22_lora_save() is False


def test_async_lora_save_gate_rejects_when_config_off(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess, async_lora_save=False)

    assert process._should_use_async_wan22_lora_save() is False


def test_async_lora_save_gate_rejects_without_network(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess, network=False)

    assert process._should_use_async_wan22_lora_save() is False


def test_save_queues_cpu_lora_snapshot_and_keeps_optimizer_sync(monkeypatch, tmp_path):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    process.save_root = str(tmp_path)
    process.job = types.SimpleNamespace(name="demo")
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
    process.accelerator = types.SimpleNamespace(is_main_process=True, is_local_main_process=True)
    process.clean_up_saves = lambda: None
    process.post_save_hook = lambda path: None
    process.update_training_metadata = lambda: None
    queue = RecordingQueue()
    process._get_async_lora_save_queue = lambda: queue

    process.save(step=10)

    assert process.network.saved_sync is False
    assert process.network.multiplier == 0.5
    assert queue.jobs[0]["output_path"].endswith("demo_000000010.safetensors")
    assert queue.jobs[0]["split_multistage_loras"] is True
    assert queue.jobs[0]["train_high_noise"] is True
    assert queue.jobs[0]["train_low_noise"] is True


def test_save_best_uses_raw_loss_and_queues_best_name(monkeypatch, tmp_path):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    process.save_config.save_best_lora = True
    process.save_root = str(tmp_path)
    process.job = types.SimpleNamespace(name="demo")
    process.name = "demo"
    process.network = FakeNetwork()
    process.embedding = None
    process.meta = {}
    process.step_num = 11
    process.epoch_num = 0
    process.sd.train_high_noise = True
    process.sd.train_low_noise = True
    process.update_training_metadata = lambda: None
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


def test_save_best_ignores_missing_loss_or_config_off(monkeypatch, tmp_path):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    process.save_root = str(tmp_path)
    process.job = types.SimpleNamespace(name="demo")
    process.name = "demo"
    process.network = FakeNetwork()
    process.embedding = None
    process.meta = {}
    process.update_training_metadata = lambda: None
    process._get_async_lora_save_queue = lambda: RecordingQueue()
    process._best_lora_loss = None
    process.save_config.save_best_lora = False

    assert process._maybe_queue_best_wan22_lora({"loss": 0.1}) is False

    process.save_config.save_best_lora = True
    assert process._maybe_queue_best_wan22_lora({}) is False


def test_cleanup_does_not_remove_best_lora_files(monkeypatch, tmp_path):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    process.save_root = str(tmp_path)
    process.job = types.SimpleNamespace(name="demo")
    process.embed_config = None
    process.sd.max_step_saves_to_keep_multiplier = 2
    process.save_config.max_step_saves_to_keep = 1
    process.accelerator = types.SimpleNamespace(is_main_process=True)

    for name in [
        "demo_000000001_high_noise.safetensors",
        "demo_000000001_low_noise.safetensors",
        "demo_best_high_noise.safetensors",
        "demo_best_low_noise.safetensors",
    ]:
        (tmp_path / name).write_text("x", encoding="utf-8")

    process.clean_up_saves()

    assert (tmp_path / "demo_best_high_noise.safetensors").exists()
    assert (tmp_path / "demo_best_low_noise.safetensors").exists()


def test_get_latest_save_path_ignores_best_lora_files(monkeypatch, tmp_path):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    process.save_root = str(tmp_path)
    process.job = types.SimpleNamespace(name="demo")
    process.network_config = types.SimpleNamespace(pretrained_lora_path=None)
    (tmp_path / "demo_best_high_noise.safetensors").write_text("x", encoding="utf-8")

    assert process.get_latest_save_path("demo") is None


def test_close_async_lora_save_queue_closes_and_clears_queue(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    queue = CloseRecordingQueue()
    process._async_lora_save_queue = queue

    process._close_async_lora_save_queue()

    assert queue.closed is True
    assert process._async_lora_save_queue is None


def test_on_error_closes_async_lora_save_queue(monkeypatch):
    module = _load_base_process_module(monkeypatch)
    process = _process_with(module.BaseSDTrainProcess)
    queue = CloseRecordingQueue()
    process._async_lora_save_queue = queue

    process.on_error(RuntimeError("boom"))

    assert queue.closed is True
    assert process._async_lora_save_queue is None
