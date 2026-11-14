"""The VoxCPM2 training child of a bake (design §8.1, PLAN §2.17 D).

Fakes only: a fake model, a fake `voxcpm.training`, a fake `save`. Nothing here imports
torch, voxcpm or datasets, and nothing loads a model.
"""

import ast
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from audio import voxcpm_trainer
from audio.voxcpm_trainer import (
    FORWARD_KEYS,
    TrainConfig,
    VoxCPMTrainer,
    adapter_metadata,
    check_trainable,
    default_steps,
    run_training,
    train_config,
)

REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
VERSION = "2.0.3"
LORA = json.dumps({"enable_dit": True, "enable_lm": True, "r": 8}, sort_keys=True)
DATASET_SHA = "d" * 64
LORA_KEYS = ("base_lm.layers.0.self_attn.q_proj.lora_A",
             "base_lm.layers.0.self_attn.q_proj.lora_B")
TEXTS = ["Hello there.", "Good morning."]
WAIT_S = 5.0
ADAPTER = b"adapter-bytes"


def test_importing_the_module_imports_no_torch_voxcpm_or_datasets():
    code = ("import sys; import audio.voxcpm_trainer; "
            "print(sorted({'torch', 'voxcpm', 'datasets'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=Path(voxcpm_trainer.__file__).parents[1])
    assert out.stdout.strip() == "[]"


def _imports(tree) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_every_import_is_torch_voxcpm_safetensors_the_hub_or_the_standard_library():
    tree = ast.parse(Path(voxcpm_trainer.__file__).read_text())
    roots = {name.split(".")[0] for name in _imports(tree)}
    allowed = {"torch", "voxcpm", "safetensors", "huggingface_hub", "numpy"}
    assert roots - allowed <= set(sys.stdlib_module_names)
    assert not roots & {"tensorboard", "accelerate", "peft", "datasets"}


def test_the_batches_never_repeat_with_itertools_cycle():
    tree = ast.parse(Path(voxcpm_trainer.__file__).read_text())
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              for alias in node.names}
    assert "cycle" not in names


def test_default_steps_are_twenty_per_take_between_200_and_1000():
    assert default_steps(1) == 200
    assert default_steps(25) == 500
    assert default_steps(80) == 1000


def test_train_config_fills_the_defaults():
    assert train_config({}, 25) == TrainConfig(steps=500, learning_rate=1e-4, batch_size=1,
                                               weight_decay=0.01, max_grad_norm=1.0, seed=42)
    assert train_config({"steps": None}, 1).steps == 200
    assert train_config({"steps": 7, "batch_size": 8}, 1).steps == 7


@pytest.mark.parametrize(("request_config", "field"), [
    ({"steps": 0}, "steps"),
    ({"steps": 5001}, "steps"),
    ({"learning_rate": 0}, "learning_rate"),
    ({"learning_rate": 0.02}, "learning_rate"),
    ({"batch_size": 9}, "batch_size"),
    ({"weight_decay": -0.1}, "weight_decay"),
    ({"max_grad_norm": 0}, "max_grad_norm"),
    ({"seed": 7}, "seed"),
    ({"epochs": 3}, "epochs"),
])
def test_train_config_refuses_a_value_out_of_range_naming_the_field(request_config, field):
    with pytest.raises(ValueError, match=field):
        train_config(request_config, 10)


def test_check_trainable_passes_lora_weights_only():
    check_trainable(["base_lm.layers.0.self_attn.q_proj.lora_A"])
    with pytest.raises(ValueError):
        check_trainable([])
    with pytest.raises(ValueError):
        check_trainable(["base_lm.layers.0.mlp.weight"])


def test_adapter_metadata_is_strings_only():
    config = TrainConfig(steps=200)
    metadata = adapter_metadata(revision=REVISION, version=VERSION, lora=LORA,
                                dataset_sha256=DATASET_SHA, config=config, steps_done=200)
    assert metadata == {
        "format": "ivms777-voxcpm2-lora-1", "model": "voxcpm2", "revision": REVISION,
        "voxcpm": VERSION, "lora": LORA, "dataset_sha256": DATASET_SHA,
        "config": json.dumps({"steps": 200, "learning_rate": 1e-4, "batch_size": 1,
                              "weight_decay": 0.01, "max_grad_norm": 1.0, "seed": 42},
                             sort_keys=True),
        "steps_done": "200",
    }


class Clock:
    """Each call is two seconds after the previous one."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += 2.0
        return self.now


def losses(*values):
    calls = []

    def train_step(batch, progress):
        calls.append((batch, progress))
        return values[len(calls) - 1]

    return train_step, calls


def endless():
    n = 0
    while True:
        n += 1
        yield f"batch-{n}"


def run(config, train_step, *, cancelled=lambda: False, memory=lambda: 0):
    reports = []
    result = run_training(config, batches=endless(), train_step=train_step,
                          cancelled=cancelled, report=reports.append, memory=memory,
                          clock=Clock())
    return result, reports


def test_run_training_runs_every_step_and_keeps_the_first_and_last_loss():
    train_step, calls = losses(3.0, 2.5, 2.0, 1.5)
    result, reports = run(TrainConfig(steps=4), train_step)
    assert result["state"] == "done"
    assert result["steps_done"] == 4
    assert (result["loss_first"], result["loss_last"]) == (3.0, 1.5)
    assert [batch for batch, _ in calls] == ["batch-1", "batch-2", "batch-3", "batch-4"]
    assert [progress for _, progress in calls] == [0.0, 0.25, 0.5, 0.75]
    assert [r["step"] for r in reports] == [1, 2, 3, 4]
    assert [r["fraction"] for r in reports] == [0.25, 0.5, 0.75, 1.0]


def test_run_training_stops_at_the_step_boundary_after_a_cancel():
    train_step, calls = losses(3.0, 2.5, 2.0, 1.5, 1.0)
    result, _ = run(TrainConfig(steps=5), train_step, cancelled=lambda: len(calls) >= 3)
    assert result["state"] == "cancelled"
    assert result["steps_done"] == 3
    assert len(calls) == 3


def test_run_training_a_cancel_in_the_last_step_ends_cancelled():
    cancel = threading.Event()

    def train_step(batch, progress):
        cancel.set()
        return 2.0

    result, reports = run(TrainConfig(steps=1), train_step, cancelled=cancel.is_set)
    assert result["state"] == "cancelled"
    assert result["steps_done"] == 1
    assert result["loss_first"] == result["loss_last"] == 2.0
    assert [r["step"] for r in reports] == [1]


def test_run_training_a_cancel_in_the_last_report_ends_cancelled():
    cancel = threading.Event()
    train_step, _ = losses(2.0)
    reports = []

    def report(progress):
        reports.append(progress)
        cancel.set()

    result = run_training(TrainConfig(steps=1), batches=endless(), train_step=train_step,
                          cancelled=cancel.is_set, report=report, memory=lambda: 0,
                          clock=Clock())
    assert result["state"] == "cancelled"
    assert result["steps_done"] == 1
    assert result["loss_last"] == 2.0
    assert [r["step"] for r in reports] == [1]


def test_run_training_fails_on_a_loss_that_is_not_a_number():
    train_step, _ = losses(3.0, math.nan)
    with pytest.raises(RuntimeError, match="not a number at step 2"):
        run(TrainConfig(steps=4), train_step)


def test_run_training_reports_the_peak_memory_and_the_seconds_per_step():
    train_step, _ = losses(3.0, 2.5, 2.0)
    readings = iter([500, 900, 700])
    result, _ = run(TrainConfig(steps=3), train_step, memory=lambda: next(readings))
    assert result["peak_device_bytes"] == 900
    # The clock moves 2 s per call: one call at the start, one after each step.
    assert result["seconds_per_step"] == pytest.approx(2.0)


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.moves = []

    def detach(self):
        self.moves.append("detach")
        return self

    def cpu(self):
        self.moves.append("cpu")
        return self

    def contiguous(self):
        self.moves.append("contiguous")
        return self


class FakeLoss:
    def __init__(self, value):
        self.value = value
        self.backward_calls = 0

    def __add__(self, other):
        return FakeLoss(self.value + other.value)

    def backward(self):
        self.backward_calls += 1

    def __float__(self):
        return float(self.value)


class FakeParameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, gate=None):
        self.device = "mps"
        self.config = "model-config"
        self.audio_vae = types.SimpleNamespace(sample_rate=16000)
        self.text_tokenizer = lambda text: [len(word) for word in text.split()]
        self.lora = FakeParameter(True)
        self.base = FakeParameter(False)
        self.forwards = []
        self.gate = gate

    def parameters(self):
        return [self.lora, self.base]

    def __call__(self, *, progress, **tensors):
        self.forwards.append((set(tensors), progress))
        if self.gate is not None:
            self.gate.entered.set()
            assert self.gate.release.wait(WAIT_S)
        return {"loss/diff": FakeLoss(2.0), "loss/stop": FakeLoss(0.5)}

    def get_lora_state_dict(self):
        return {name: FakeTensor(name) for name in LORA_KEYS}


class Gate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()


class FakeDataset:
    def __init__(self):
        self.columns = {"text": list(TEXTS)}
        self.added = []
        self.mapped = 0

    def __getitem__(self, column):
        return self.columns[column]

    def add_column(self, name, values):
        self.added.append((name, values))
        self.columns[name] = values
        return self

    def map(self, *args, **kwargs):
        self.mapped += 1
        return self


class FakeLoader:
    """Two batches a pass; counts how often a pass starts."""

    def __init__(self):
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter([{"batch": 1}, {"batch": 2}])


class FakeTraining:
    """Stands in for `voxcpm.training`."""

    def __init__(self, fail=None):
        self.dataset = FakeDataset()
        self.loader = FakeLoader()
        self.fail = fail
        self.calls = []

    def load_audio_text_datasets(self, *, train_manifest, sample_rate):
        self.calls.append(("load", train_manifest, sample_rate))
        if self.fail is not None:
            raise self.fail
        return self.dataset, None

    def build_dataloader(self, ds, *, accelerator, batch_size, num_workers):
        self.calls.append(("loader", ds, accelerator.device, batch_size, num_workers))
        return self.loader

    def BatchProcessor(self, *, config, audio_vae, dataset_cnt, device):
        self.calls.append(("processor", config, dataset_cnt, device))
        return lambda batch: {**{key: batch["batch"] for key in FORWARD_KEYS},
                              "audio_task_ids": 0}


class FakeOptimizer:
    def __init__(self, params, *, lr, weight_decay):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps = 0
        self.zeroed = []

    def step(self):
        self.steps += 1

    def zero_grad(self, *, set_to_none):
        self.zeroed.append(set_to_none)


class Saves:
    def __init__(self):
        self.calls = []

    def __call__(self, tensors, path, *, metadata):
        self.calls.append((tensors, path, metadata))
        Path(path).write_bytes(ADAPTER)


def trainer_with(model=None, training=None, save=None):
    optimizers = []
    clips = []

    def optimizer(params, **kwargs):
        optimizers.append(FakeOptimizer(params, **kwargs))
        return optimizers[-1]

    trainer = VoxCPMTrainer.loaded(
        model or FakeModel(), REVISION, lora=LORA, data=training or FakeTraining(),
        optimizer=optimizer, clip=lambda params, norm: clips.append((params, norm)),
        save=save or Saves(), memory=lambda: 4096, version=VERSION,
    )
    return trainer, optimizers, clips


def job(tmp_path, **config):
    return {"manifest": str(tmp_path / "train.jsonl"), "work_dir": str(tmp_path),
            "clips": 2, "config": config, "dataset_sha256": DATASET_SHA}


def wait_for_end(trainer) -> dict:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        status = trainer.status()
        if status["state"] not in ("idle", "running"):
            return status
        time.sleep(0.01)
    raise AssertionError(f"the trainer did not end: {trainer.status()}")


def test_a_fresh_trainer_is_idle():
    trainer, _, _ = trainer_with()
    assert trainer.status() == {
        "state": "idle", "stage": None, "step": 0, "steps": 0, "loss_first": None,
        "loss_last": None, "seconds_per_step": None, "peak_device_bytes": None,
        "device": "mps", "error": None, "adapter": None,
    }


def test_start_returns_at_once_and_the_thread_trains_to_done(tmp_path):
    model = FakeModel()
    training = FakeTraining()
    saves = Saves()
    trainer, optimizers, clips = trainer_with(model, training, saves)
    trainer.start(job(tmp_path, steps=3, learning_rate=2e-4))
    status = wait_for_end(trainer)

    assert status["state"] == "done", status
    assert (status["step"], status["steps"]) == (3, 3)
    assert (status["loss_first"], status["loss_last"]) == (2.5, 2.5)
    assert status["peak_device_bytes"] == 4096
    assert status["error"] is None
    assert [progress for _, progress in model.forwards] == [0.0, 1 / 3, 2 / 3]
    assert all(keys == set(FORWARD_KEYS) for keys, _ in model.forwards)
    [optimizer] = optimizers
    assert optimizer.params == [model.lora]
    assert (optimizer.lr, optimizer.weight_decay) == (2e-4, 0.01)
    assert (optimizer.steps, optimizer.zeroed) == (3, [True, True, True])
    assert clips == [([model.lora], 1.0)] * 3
    assert training.calls[0] == ("load", str(tmp_path / "train.jsonl"), 16000)
    assert training.calls[1][2:] == ("mps", 1, 0)
    assert training.calls[2] == ("processor", "model-config", 1, "mps")


def test_the_text_ids_are_added_as_a_column_from_the_model_tokenizer(tmp_path):
    training = FakeTraining()
    trainer, _, _ = trainer_with(training=training)
    trainer.start(job(tmp_path, steps=1))
    wait_for_end(trainer)
    assert training.dataset.added == [("text_ids", [[5, 6], [4, 8]])]
    assert training.dataset.mapped == 0


def test_the_batches_start_a_second_pass_over_the_loader(tmp_path):
    training = FakeTraining()
    trainer, _, _ = trainer_with(training=training)
    trainer.start(job(tmp_path, steps=3))
    assert wait_for_end(trainer)["state"] == "done"
    assert training.loader.passes == 2


def test_save_gets_every_lora_tensor_on_the_cpu_and_the_metadata(tmp_path):
    saves = Saves()
    trainer, _, _ = trainer_with(save=saves)
    trainer.start(job(tmp_path, steps=2))
    status = wait_for_end(trainer)

    [(tensors, path, metadata)] = saves.calls
    assert set(tensors) == set(LORA_KEYS)
    assert all(t.moves == ["detach", "cpu", "contiguous"] for t in tensors.values())
    assert path == str(tmp_path / "adapter.safetensors")
    assert metadata["format"] == "ivms777-voxcpm2-lora-1"
    assert metadata["model"] == "voxcpm2"
    assert metadata["revision"] == REVISION
    assert metadata["voxcpm"] == VERSION
    assert metadata["lora"] == LORA
    assert metadata["dataset_sha256"] == DATASET_SHA
    assert json.loads(metadata["config"])["steps"] == 2
    assert metadata["steps_done"] == "2"
    assert status["adapter"] == {"path": path, "sha256": hashlib.sha256(ADAPTER).hexdigest(),
                                 "bytes": len(ADAPTER), "keys": 2}


def test_status_answers_while_a_step_runs(tmp_path):
    gate = Gate()
    trainer, _, _ = trainer_with(FakeModel(gate))
    trainer.start(job(tmp_path, steps=1))
    assert gate.entered.wait(WAIT_S)

    status = trainer.status()
    assert (status["state"], status["stage"], status["step"]) == ("running", "training", 0)
    gate.release.set()
    assert wait_for_end(trainer)["state"] == "done"


def test_a_second_start_is_refused(tmp_path):
    trainer, _, _ = trainer_with()
    trainer.start(job(tmp_path, steps=1))
    with pytest.raises(RuntimeError):
        trainer.start(job(tmp_path, steps=1))
    wait_for_end(trainer)
    with pytest.raises(RuntimeError):
        trainer.start(job(tmp_path, steps=1))


@pytest.mark.parametrize("steps", [1, 5])
def test_a_cancel_during_training_ends_cancelled_and_saves_nothing(tmp_path, steps):
    gate = Gate()
    saves = Saves()
    model = FakeModel(gate)
    trainer, _, _ = trainer_with(model, save=saves)
    trainer.start(job(tmp_path, steps=steps))
    try:
        assert gate.entered.wait(WAIT_S)
        trainer.cancel()
    finally:
        gate.release.set()
    status = wait_for_end(trainer)
    assert status["state"] == "cancelled"
    assert status["step"] == 1
    assert len(model.forwards) == 1
    assert saves.calls == []
    assert status["adapter"] is None
    assert list(tmp_path.rglob("adapter.safetensors")) == []


def test_an_exception_in_the_data_step_fails_with_its_type_and_message(tmp_path):
    saves = Saves()
    trainer, _, _ = trainer_with(training=FakeTraining(fail=ValueError("bad manifest")),
                                 save=saves)
    trainer.start(job(tmp_path, steps=1))
    status = wait_for_end(trainer)
    assert status["state"] == "failed"
    assert status["error"] == "ValueError: bad manifest"
    assert saves.calls == []


def test_a_cpu_device_is_refused_before_any_import(monkeypatch):
    for name in ("huggingface_hub", "voxcpm", "voxcpm.training", "torch", "datasets"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match="mps or cuda only"):
        VoxCPMTrainer("repo", "cpu", "rev", "{}", "/tmp")


class FakeImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves `voxcpm.training` and `datasets`, and records the HF env at each import.

    The fake `voxcpm.training` imports `datasets` first, as the real one does
    (`training/data.py:6`).
    """

    def __init__(self, accelerator_device):
        self.accelerator_device = accelerator_device
        self.seen = []

    def find_spec(self, fullname, path, target=None):
        if fullname in ("voxcpm.training", "datasets"):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.seen.append((module.__name__, os.environ.get("HF_DATASETS_CACHE"),
                          os.environ.get("HF_DATASETS_OFFLINE")))
        if module.__name__ == "voxcpm.training":
            import datasets  # noqa: F401 - the fake one, served by this finder

            device = types.SimpleNamespace(type=self.accelerator_device)
            module.Accelerator = lambda amp, seed: types.SimpleNamespace(device=device)


@pytest.fixture
def fake_imports(monkeypatch, tmp_path):
    def install(accelerator_device):
        snapshot = tmp_path / "snapshots" / REVISION
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda repo, revision, local_files_only: str(snapshot)
        errors = types.ModuleType("huggingface_hub.errors")
        errors.LocalEntryNotFoundError = type("LocalEntryNotFoundError", (Exception,), {})
        package = types.ModuleType("voxcpm")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
        monkeypatch.setitem(sys.modules, "voxcpm", package)
        monkeypatch.delitem(sys.modules, "voxcpm.training", raising=False)
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
        monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
        finder = FakeImports(accelerator_device)
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        return finder

    yield install
    for name in ("voxcpm.training", "datasets"):
        sys.modules.pop(name, None)


def test_an_accelerator_on_another_device_is_refused(fake_imports, tmp_path):
    fake_imports("cpu")
    with pytest.raises(RuntimeError, match="training device is 'cpu', not 'mps'"):
        VoxCPMTrainer("openbmb/VoxCPM2", "mps", REVISION, LORA, str(tmp_path))


def test_the_datasets_env_is_set_before_datasets_is_imported(fake_imports, tmp_path):
    finder = fake_imports("cpu")
    with pytest.raises(RuntimeError, match="training device"):
        VoxCPMTrainer("openbmb/VoxCPM2", "mps", REVISION, LORA, str(tmp_path))
    cache = f"{tmp_path}/hf-cache"
    assert finder.seen == [("voxcpm.training", cache, "1"), ("datasets", cache, "1")]
