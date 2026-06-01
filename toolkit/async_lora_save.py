import os
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional

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
        self._worker = threading.Thread(
            target=self._run,
            name="aitk-lora-save",
            daemon=False,
        )
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
