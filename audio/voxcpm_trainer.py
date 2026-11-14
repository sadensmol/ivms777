"""The VoxCPM2 training child of a bake: one LoRA adapter from a saved voice's takes
(design §5.1, §8.1; docs/models.md "Voice baking").

Models-service-only, like every other module in `audio/`. The `voice_train` unit hosts
it in a `TorchWorker` child. One call to that child holds the worker's lock until it
returns, so `start` trains on a thread of its own and returns at once; `status` and
`cancel` answer between steps.

Module scope imports no torch, no voxcpm and no datasets, exactly like
`audio/voxcpm_voice.py`: the pure parts (`train_config`, `check_trainable`,
`adapter_metadata`, `run_training`) are unit-tested, and importing them must not build
a torch context in the importing process (CLAUDE.md § "One model process").

voxcpm 2.0.3 ships `voxcpm.training` but no training script and no config, so the
loop is ours; its defaults are the planner's (D58) until the user's bake measures them.
"""

import hashlib
import json
import math
import os
import threading
import time
import types
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_KEY = "voxcpm2"
ADAPTER_FORMAT = "ivms777-voxcpm2-lora-1"
ADAPTER_FILE = "adapter.safetensors"
# The package Accelerator's own default (`training/accelerator.py:23`). The model build
# draws every LoRA A from it, so a request cannot change it.
SEED = 42
# The tensor arguments of `VoxCPM2Model.forward` (`voxcpm2.py:297-306`).
FORWARD_KEYS = ("text_tokens", "text_mask", "audio_feats", "audio_mask", "loss_mask",
                "position_ids", "labels")


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    learning_rate: float = 1e-4
    batch_size: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = SEED


def default_steps(clips: int) -> int:
    """About 20 passes over each take, at least 200 steps and at most 1000."""
    return min(1000, max(200, 20 * clips))


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def train_config(request: dict, clips: int) -> TrainConfig:
    """The loop values of a bake request, each checked; `ValueError` names the field."""
    known = {field.name for field in fields(TrainConfig)}
    for name in request:
        if name not in known:
            raise ValueError(f"config has an unknown field {name!r}")
    config = TrainConfig(**{"steps": None, **request})
    steps = default_steps(clips) if config.steps is None else config.steps
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 5000:
        raise ValueError(f"steps must be 1-5000, not {steps!r}")
    lr = config.learning_rate
    if not _number(lr) or not 0 < lr <= 1e-2:
        raise ValueError(f"learning_rate must be above 0 and at most 0.01, not {lr!r}")
    batch = config.batch_size
    if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 8:
        raise ValueError(f"batch_size must be 1-8, not {batch!r}")
    wd = config.weight_decay
    if not _number(wd) or not 0 <= wd <= 1:
        raise ValueError(f"weight_decay must be 0-1, not {wd!r}")
    norm = config.max_grad_norm
    if not _number(norm) or not 0 < norm <= 100:
        raise ValueError(f"max_grad_norm must be above 0 and at most 100, not {norm!r}")
    if config.seed != SEED:
        raise ValueError(f"seed is fixed at {SEED}, not {config.seed!r}")
    return TrainConfig(steps=steps, learning_rate=float(lr), batch_size=batch,
                       weight_decay=float(wd), max_grad_norm=float(norm))


def check_trainable(names: list[str]) -> None:
    """Refuse a model that would train anything but its LoRA weights."""
    if not names:
        raise ValueError("the model has no trainable LoRA weight")
    other = [name for name in names if "lora_" not in name]
    if other:
        raise ValueError(f"{len(other)} trainable weights are not LoRA weights, "
                         f"first {other[0]!r}")


def adapter_metadata(*, revision, version, lora: str, dataset_sha256, config: TrainConfig,
                     steps_done) -> dict[str, str]:
    """The adapter file's metadata; safetensors stores strings only."""
    return {
        "format": ADAPTER_FORMAT,
        "model": MODEL_KEY,
        "revision": str(revision),
        "voxcpm": str(version),
        "lora": lora,
        "dataset_sha256": str(dataset_sha256),
        "config": json.dumps(asdict(config), sort_keys=True),
        "steps_done": str(steps_done),
    }


def run_training(config, *, batches, train_step, cancelled, report, memory, clock) -> dict:
    """`config.steps` steps of `train_step(batch, progress)`, stopping at the first step
    boundary where `cancelled()` is true.

    `report` gets the progress after each step. A loss that is not a number fails the
    bake, because every later step would train on it.
    """
    loss_first = loss_last = seconds_per_step = peak = None
    steps_done = 0
    started = clock()
    state = "done"
    for step in range(1, config.steps + 1):
        if cancelled():
            state = "cancelled"
            break
        loss = train_step(next(batches), (step - 1) / config.steps)
        if not math.isfinite(loss):
            raise RuntimeError(f"the training loss is not a number at step {step}")
        steps_done = step
        loss_last = loss
        loss_first = loss if loss_first is None else loss_first
        used = memory()
        peak = used if peak is None else max(peak, used)
        seconds_per_step = (clock() - started) / step
        report({"step": step, "fraction": step / config.steps, "loss_first": loss_first,
                "loss_last": loss_last, "seconds_per_step": seconds_per_step,
                "peak_device_bytes": peak})
    # The end of the last step is a step boundary too.
    if state == "done" and cancelled():
        state = "cancelled"
    return {"state": state, "steps_done": steps_done, "loss_first": loss_first,
            "loss_last": loss_last, "seconds_per_step": seconds_per_step,
            "peak_device_bytes": peak}


def _voxcpm_version() -> str:
    from importlib.metadata import version

    return version("voxcpm")


class VoxCPMTrainer:
    def __init__(self, repo: str, device: str, revision: str, lora: str, bake_root: str) -> None:
        # GPU only (`docs/design.md` §5), as the voice child (`audio/voxcpm_voice.py`).
        if device != "mps" and not str(device).startswith("cuda"):
            raise RuntimeError(f"VoxCPM2 trains on mps or cuda only, not {device!r}")

        # `datasets` reads both when it is imported (by `voxcpm.training`): the takes'
        # cache stays in the bake folder, and nothing is fetched from the Hub.
        os.environ["HF_DATASETS_CACHE"] = f"{bake_root}/hf-cache"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"VoxCPM2 weights for {revision} are not on disk: run `make models`, "
                "or POST /models/download {slot: voice, key: voxcpm2}"
            ) from exc
        if Path(path).name != revision:
            raise RuntimeError(
                f"VoxCPM2 snapshot {Path(path).name!r} is not the pinned revision {revision!r}"
            )

        from voxcpm import training

        # Built before the model, so its seed fixes the LoRA A draw of the build.
        accelerator = training.Accelerator(amp=False, seed=SEED)
        # The package falls back to the CPU with no warning (`accelerator.py:106-112`).
        if accelerator.device.type != device.split(":")[0]:
            raise RuntimeError(
                f"the training device is {accelerator.device.type!r}, not {device!r}"
            )

        import torch
        from safetensors.torch import save_file
        from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model

        model = VoxCPM2Model.from_local(path, optimize=False, training=True, device=device,
                                        lora_config=LoRAConfig(**json.loads(lora)))
        model = accelerator.prepare_model(model)
        model.train()
        check_trainable([name for name, p in model.named_parameters() if p.requires_grad])
        memory = (torch.mps.driver_allocated_memory if device == "mps"
                  else torch.cuda.max_memory_allocated)
        self._setup(model, revision, lora, accelerator, training, torch.optim.AdamW,
                    torch.nn.utils.clip_grad_norm_, save_file, memory, _voxcpm_version())

    @classmethod
    def loaded(cls, model, revision: str, *, lora: str, data, optimizer, clip, save, memory,
               version="test") -> "VoxCPMTrainer":
        """A child around an already built model — the test seam.

        `data` stands in for `voxcpm.training`; `optimizer`, `clip`, `save` and `memory`
        for `torch.optim.AdamW`, `clip_grad_norm_`, `safetensors.torch.save_file` and the
        device's memory counter. The accelerator is only its device, the model's own
        (`prepare_model` sets it, `accelerator.py:84-86`).
        """
        trainer = cls.__new__(cls)
        accelerator = types.SimpleNamespace(device=model.device)
        trainer._setup(model, revision, lora, accelerator, data, optimizer, clip, save,
                       memory, version)
        return trainer

    def _setup(self, model, revision, lora, accelerator, data, optimizer, clip, save, memory,
               version) -> None:
        self._model = model
        self.revision = revision
        self._lora = lora
        self._accelerator = accelerator
        self._data = data
        self._optimizer = optimizer
        self._clip = clip
        self._save = save
        self._memory = memory
        self._version = version
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._status = {
            "state": "idle", "stage": None, "step": 0, "steps": 0, "loss_first": None,
            "loss_last": None, "seconds_per_step": None, "peak_device_bytes": None,
            "device": str(accelerator.device), "error": None, "adapter": None,
        }

    def start(self, job: dict) -> None:
        """Train on a daemon thread and return at once.

        `job` = `{"manifest", "work_dir", "clips", "config", "dataset_sha256"}`; `clips`
        is the number of takes and `config` the request's loop values (`train_config`).
        One child trains once: the bake unloads it at the end.
        """
        config = train_config(job["config"], job["clips"])
        with self._lock:
            if self._status["state"] != "idle":
                raise RuntimeError("this trainer has already started")
            self._status.update(state="running", stage="reading the takes",
                                steps=config.steps)
        threading.Thread(target=self._train, args=(job, config), daemon=True,
                         name="voxcpm-train").start()

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def cancel(self) -> None:
        self._cancel.set()

    def _update(self, **values) -> None:
        with self._lock:
            self._status.update(values)

    def _report(self, progress: dict) -> None:
        # The parent turns the step into its own fraction of the whole bake.
        self._update(**{key: value for key, value in progress.items() if key != "fraction"})

    def _train(self, job: dict, config: TrainConfig) -> None:
        try:
            result = self._run(job, config)
        except Exception as exc:  # noqa: BLE001 - the status carries it to the parent
            self._update(state="failed", error=f"{type(exc).__name__}: {exc}")
            return
        self._update(**result)

    def _run(self, job: dict, config: TrainConfig) -> dict:
        model = self._model
        data = self._data
        ds, _ = data.load_audio_text_datasets(train_manifest=job["manifest"],
                                              sample_rate=model.audio_vae.sample_rate)
        # Not `ds.map`: `datasets` fingerprints a map transform by pickling it, and a
        # closure over the model would copy every tensor of it (§1.8).
        tok = model.text_tokenizer
        ds = ds.add_column("text_ids", [list(tok(text)) for text in ds["text"]])
        loader = data.build_dataloader(ds, accelerator=self._accelerator,
                                       batch_size=config.batch_size, num_workers=0)

        def batches():
            # Each pass asks the loader again, so it shuffles again; `itertools.cycle`
            # would replay the first pass's batches.
            while True:
                yield from loader

        processor = data.BatchProcessor(config=model.config, audio_vae=model.audio_vae,
                                        dataset_cnt=1, device=self._accelerator.device)

        self._update(stage="training")
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._optimizer(trainable, lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

        def train_step(batch, progress: float) -> float:
            packed = processor(batch)
            out = model(**{key: packed[key] for key in FORWARD_KEYS}, progress=progress)
            loss = out["loss/diff"] + out["loss/stop"]
            loss.backward()
            self._clip(trainable, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return float(loss)

        result = run_training(
            config, batches=batches(), train_step=train_step, cancelled=self._cancel.is_set,
            report=self._report, memory=self._memory, clock=time.monotonic,
        )
        measured = {key: result[key] for key in ("loss_first", "loss_last",
                                                  "seconds_per_step", "peak_device_bytes")}
        if result["state"] == "cancelled" or self._cancel.is_set():
            return {"state": "cancelled", **measured}

        self._update(stage="saving the voice model")
        tensors = {key: value.detach().cpu().contiguous()
                   for key, value in model.get_lora_state_dict().items()}
        path = Path(job["work_dir"]) / ADAPTER_FILE
        self._save(tensors, str(path), metadata=adapter_metadata(
            revision=self.revision, version=self._version, lora=self._lora,
            dataset_sha256=job["dataset_sha256"], config=config,
            steps_done=result["steps_done"],
        ))
        adapter = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "bytes": path.stat().st_size, "keys": len(tensors)}
        return {"state": "done", "adapter": adapter, **measured}
