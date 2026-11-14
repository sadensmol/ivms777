"""The VoxCPM2 voice-copy child (design §11.1, PLAN §2.5).

Fakes only: a fake `tts_model`, pickle in place of `torch.save` / `torch.load`, and a
fake `voxcpm` module. Nothing here imports torch or loads a model.
"""

import ast
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from audio import voxcpm_voice
from audio.voxcpm_voice import (
    CONTROLS,
    VoxCPMVoice,
    adapter_mismatch,
    check_references,
    load_mismatch,
    payload_mismatch,
    pick_conditioning,
    tensor_mismatch,
)
from storage.wav import decode_wav

REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
VERSION = "2.0.3"


class FakeWav:
    """The `(1, n)` CPU tensor `generate_with_prompt_cache` returns."""

    def __init__(self, samples):
        self._samples = np.asarray(samples, dtype=np.float32).reshape(1, -1)

    def squeeze(self, dim):
        return FakeWav(self._samples) if dim else _Squeezed(self._samples[0])


class _Squeezed:
    def __init__(self, samples):
        self._samples = samples

    def cpu(self):
        return self

    def numpy(self):
        return self._samples


class FakeTTS:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.built: list[dict] = []
        self.calls: list[dict] = []

    def build_prompt_cache(self, prompt_text=None, prompt_wav_path=None,
                           reference_wav_path=None):
        # The package's own mode rule (`voxcpm2.py:731-738`).
        clip = Path(reference_wav_path).read_bytes()
        self.built.append({"path": reference_wav_path, "clip": clip,
                           "prompt_text": prompt_text, "prompt_wav_path": prompt_wav_path})
        if prompt_wav_path is None:
            return {"ref_audio_feat": clip, "mode": "reference"}
        return {"ref_audio_feat": clip, "prompt_text": prompt_text,
                "audio_feat": Path(prompt_wav_path).read_bytes(),
                "mode": "ref_continuation"}

    def generate_with_prompt_cache(self, **kwargs):
        self.calls.append(kwargs)
        # 10 samples per character, so the output length is checkable.
        return FakeWav(np.full(len(kwargs["target_text"]) * 10, 0.1)), None, None


def ref(clip: bytes, conditioning=None, transcript=None) -> dict:
    return {
        "audio": clip, "sha256": hashlib.sha256(clip).hexdigest(), "language": "en",
        "transcript": transcript, "conditioning": conditioning or [],
    }


def child(tts=None, version=VERSION) -> VoxCPMVoice:
    return VoxCPMVoice.loaded(tts or FakeTTS(), REVISION, save=pickle.dumps,
                              load=pickle.loads, version=version)


@pytest.fixture
def fake_voxcpm(monkeypatch):
    """A fake `voxcpm` package and a fixed version, so `__init__` runs with no torch."""
    calls = []

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            model = cls()
            model.tts_model = FakeTTS()
            return model

    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=FakeVoxCPM))
    monkeypatch.setattr(voxcpm_voice, "_voxcpm_version", lambda: VERSION)
    return calls


class LocalEntryNotFoundError(Exception):
    """Stands in for `huggingface_hub.errors.LocalEntryNotFoundError`: the Hub client
    is in the `models` extra only, and a core-only test run must not need it."""


def snapshot_at(monkeypatch, folder: str | None):
    calls = []

    def snapshot_download(repo, revision=None, local_files_only=False):
        calls.append((repo, revision, local_files_only))
        if folder is None:
            raise LocalEntryNotFoundError("not cached")
        return f"/hf/models--openbmb--VoxCPM2/snapshots/{folder}"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return calls


def test_cpu_is_refused(monkeypatch, fake_voxcpm):
    snapshot_at(monkeypatch, REVISION)
    with pytest.raises(RuntimeError, match="mps or cuda only"):
        VoxCPMVoice("openbmb/VoxCPM2", "cpu", REVISION)
    assert fake_voxcpm == []


def test_a_snapshot_of_another_revision_is_refused(monkeypatch, fake_voxcpm):
    snapshot_at(monkeypatch, "0" * 40)
    with pytest.raises(RuntimeError, match="not the pinned revision"):
        VoxCPMVoice("openbmb/VoxCPM2", "mps", REVISION)
    assert fake_voxcpm == []


def test_missing_weights_name_the_fix(monkeypatch, fake_voxcpm):
    snapshot_at(monkeypatch, None)
    with pytest.raises(RuntimeError) as raised:
        VoxCPMVoice("openbmb/VoxCPM2", "cuda", REVISION)
    message = str(raised.value)
    assert REVISION in message and "make models" in message
    assert "POST /models/download {slot: voice, key: voxcpm2}" in message


def test_it_loads_the_pinned_local_snapshot_without_denoiser_or_compile(monkeypatch, fake_voxcpm):
    downloads = snapshot_at(monkeypatch, REVISION)
    voice = VoxCPMVoice("openbmb/VoxCPM2", "cuda:0", REVISION)
    assert downloads == [("openbmb/VoxCPM2", REVISION, True)]
    assert fake_voxcpm == [(
        f"/hf/models--openbmb--VoxCPM2/snapshots/{REVISION}",
        {"load_denoiser": False, "optimize": False, "device": "cuda:0"},
    )]
    assert voice.revision == REVISION
    assert voice.sample_rate == 48000


def test_warm_is_one_short_zero_shot_call():
    tts = FakeTTS()
    child(tts).warm()
    assert tts.calls == [{"target_text": "Ready.", "prompt_cache": None, "max_len": 64,
                          "retry_badcase": False}]


def test_built_then_restored():
    tts = FakeTTS()
    clip = b"RIFF-example-one"
    first = child(tts).speak("Hello there.", "saved:1", references={"saved:1": ref(clip)})
    built = first["conditioning"]["saved:1"]
    assert built["state"] == "built"
    assert built["reason"] == "no stored conditioning"
    assert (built["model"], built["revision"]) == ("voxcpm2", REVISION)
    # The clip reached the model through a temporary file that is gone afterwards.
    assert tts.built[0]["clip"] == clip
    assert not os.path.exists(tts.built[0]["path"])

    stored = [{"model": "voxcpm2", "revision": REVISION, "blob": built["blob"]}]
    tts_again = FakeTTS()
    second = child(tts_again).speak("Hello there.", "saved:1",
                                    references={"saved:1": ref(clip, stored)})
    restored = second["conditioning"]["saved:1"]
    assert (restored["state"], restored["blob"], restored["reason"]) == ("restored", None, None)
    assert tts_again.built == []
    assert tts_again.calls[0]["prompt_cache"] == {"ref_audio_feat": clip, "mode": "reference"}


def blob_for(clip: bytes, *, version=VERSION) -> bytes:
    """A real built blob of `clip`, as a child with voxcpm `version` made it."""
    reply = child(version=version).speak("Hi.", "t", references={"t": ref(clip)})
    return reply["conditioning"]["t"]["blob"]


@pytest.mark.parametrize(("stored", "reason"), [
    ([{"model": "voxcpm2", "revision": "1" * 40, "blob": b"x"}], "stored for another revision"),
    ([{"model": "other-model", "revision": REVISION, "blob": b"x"}], "no stored conditioning"),
    ([{"model": "voxcpm2", "revision": REVISION, "blob": "OLD_VERSION"}], "voxcpm version changed"),
    ([{"model": "voxcpm2", "revision": REVISION, "blob": "OTHER_CLIP"}], "clip changed"),
    ([{"model": "voxcpm2", "revision": REVISION, "blob": b"not a pickle"}],
     "stored conditioning is invalid"),
])
def test_a_stale_blob_is_rebuilt_with_its_reason(stored, reason):
    clip = b"RIFF-current-clip"
    if stored[0]["blob"] == "OLD_VERSION":
        stored[0]["blob"] = blob_for(clip, version="2.0.2")
    elif stored[0]["blob"] == "OTHER_CLIP":
        stored[0]["blob"] = blob_for(b"RIFF-an-older-clip")
    tts = FakeTTS()
    reply = child(tts).speak("Hi.", "t", references={"t": ref(clip, stored)})
    entry = reply["conditioning"]["t"]
    assert (entry["state"], entry["reason"]) == ("built", reason)
    assert pickle.loads(entry["blob"]) == {
        "format": 1, "model": "voxcpm2", "revision": REVISION, "voxcpm": VERSION,
        "sha256": hashlib.sha256(clip).hexdigest(), "mode": "reference",
        "transcript": None, "cache": {"ref_audio_feat": clip, "mode": "reference"},
    }
    assert len(tts.built) == 1


def test_a_recorded_example_is_cloned_with_its_transcript():
    # Reference mode copies the TIMBRE only; the pace came out a third faster than the
    # person's own clip. The clip plus its transcript continues from their own speech,
    # which keeps "timbre, rhythm, emotion, and style" (the package's Ultimate Cloning).
    tts = FakeTTS()
    clip = b"RIFF-recorded"
    reply = child(tts).speak("Hello there.", "t", references={
        "t": ref(clip, transcript="Morning light slid across the table.")})
    assert tts.built[0]["prompt_text"] == "Morning light slid across the table."
    assert tts.built[0]["prompt_wav_path"] == tts.built[0]["path"]
    assert tts.calls[0]["prompt_cache"]["mode"] == "ref_continuation"
    payload = pickle.loads(reply["conditioning"]["t"]["blob"])
    assert payload["mode"] == "ref_continuation"
    assert payload["transcript"] == "Morning light slid across the table."


def test_an_example_without_a_transcript_stays_in_reference_mode():
    tts = FakeTTS()
    child(tts).speak("Hi.", "t", references={"t": ref(b"RIFF-uploaded", transcript="  ")})
    assert tts.built[0]["prompt_wav_path"] is None
    assert tts.calls[0]["prompt_cache"]["mode"] == "reference"


def test_a_reference_mode_blob_is_rebuilt_once_the_transcript_is_used():
    clip = b"RIFF-recorded"
    stored = [{"model": "voxcpm2", "revision": REVISION, "blob": blob_for(clip)}]
    reply = child().speak("Hi.", "t", references={
        "t": ref(clip, stored, transcript="The words.")})
    entry = reply["conditioning"]["t"]
    assert (entry["state"], entry["reason"]) == ("built", "cloning mode changed")


def test_a_blob_built_from_another_transcript_is_rebuilt():
    clip = b"RIFF-recorded"
    old = child().speak("Hi.", "t", references={"t": ref(clip, transcript="Old words.")})
    stored = [{"model": "voxcpm2", "revision": REVISION,
               "blob": old["conditioning"]["t"]["blob"]}]
    reply = child().speak("Hi.", "t", references={
        "t": ref(clip, stored, transcript="New words.")})
    entry = reply["conditioning"]["t"]
    assert (entry["state"], entry["reason"]) == ("built", "transcript changed")


def test_pick_conditioning_takes_the_newest_match():
    stored = [
        {"model": "voxcpm2", "revision": REVISION, "blob": b"old"},
        {"model": "voxcpm2", "revision": "1" * 40, "blob": b"other"},
        {"model": "voxcpm2", "revision": REVISION, "blob": b"new"},
    ]
    assert pick_conditioning(ref(b"c", stored), "voxcpm2", REVISION) == (b"new", None)


def test_payload_mismatch_accepts_only_an_exact_payload():
    good = {"format": 1, "model": "voxcpm2", "revision": REVISION, "voxcpm": VERSION,
            "sha256": "ab", "mode": "reference", "transcript": None, "cache": {}}
    assert payload_mismatch(good, REVISION, VERSION, "ab", None) is None
    assert payload_mismatch({**good, "mode": "continuation"}, REVISION, VERSION, "ab",
                            None) == "stored conditioning is invalid"
    assert payload_mismatch(good, REVISION, VERSION, "ab", "words") == (
        "cloning mode changed")
    assert payload_mismatch(None, REVISION, VERSION, "ab", None) == (
        "stored conditioning is invalid")


def test_an_unknown_token_is_refused():
    references = {"saved:1": ref(b"one")}
    with pytest.raises(ValueError, match="saved:2"):
        child().speak("Hi.", "saved:2", references=references)
    with pytest.raises(ValueError, match="saved:9"):
        child().speak("A. B.", "saved:1", segments=["A.", "B."], voices=[None, "saved:9"],
                      references=references)
    with pytest.raises(ValueError, match="no references"):
        child().speak("Hi.", "af_heart")


def test_check_references_lists_each_used_token_once():
    references = {"a": {}, "b": {}}
    assert check_references("a", [None, "b", "a", ""], references) == ["a", "b"]


def test_each_passage_speaks_with_its_own_tokens_cache():
    tts = FakeTTS()
    references = {"saved:1": ref(b"clip-one"), "saved:2": ref(b"clip-two")}
    reply = child(tts).speak(
        "One. Two. Three.", "saved:1", segments=["One.", "Two.", "Three."],
        voices=[None, "saved:2", None], references=references,
    )
    used = [(call["target_text"], call["prompt_cache"]["ref_audio_feat"]) for call in tts.calls]
    assert used == [("One.", b"clip-one"), ("Two.", b"clip-two"), ("Three.", b"clip-one")]
    assert set(reply["conditioning"]) == {"saved:1", "saved:2"}
    assert len(tts.built) == 2


def test_the_package_generate_controls_are_passed():
    tts = FakeTTS()
    child(tts).speak("Hello\n  there.", "t", references={"t": ref(b"c")})
    call = tts.calls[0]
    assert call["target_text"] == "Hello there."
    assert {k: v for k, v in call.items() if k not in ("target_text", "prompt_cache")} == CONTROLS
    assert CONTROLS == {"min_len": 2, "max_len": 4096, "inference_timesteps": 10,
                        "cfg_value": 2.0, "retry_badcase": True, "retry_badcase_max_times": 3,
                        "retry_badcase_ratio_threshold": 6.0}


def test_a_tokens_controls_cover_only_its_own_runs_and_leave_controls_unchanged():
    tts = FakeTTS()
    before = dict(CONTROLS)
    set_by_test = {"cfg_value": 3.0, "inference_timesteps": 20}
    child(tts).speak(
        "One. Two. Three.", "plain", segments=["One.", "Two.", "Three."],
        voices=[None, "swept", None],
        references={"plain": ref(b"p"), "swept": ref(b"s") | {"controls": set_by_test}})
    used = [{k: v for k, v in call.items() if k not in ("target_text", "prompt_cache")}
            for call in tts.calls]
    assert used == [before, before | set_by_test, before]
    assert CONTROLS == before


def test_the_reply_echoes_controls_only_for_the_tokens_that_set_them():
    set_by_test = {"cfg_value": 1.5, "inference_timesteps": 20}
    reply = child().speak(
        "One. Two.", "plain", segments=["One.", "Two."], voices=[None, "swept"],
        references={"plain": ref(b"p"), "swept": ref(b"s") | {"controls": set_by_test}})
    assert reply["controls"] == {"swept": set_by_test}


def test_a_long_passage_is_spoken_in_pieces_of_at_most_500_characters():
    tts = FakeTTS()
    text = "We drove along the coast in the rain. " * 60
    reply = child(tts).speak(text, "t", references={"t": ref(b"c")})
    pieces = [call["target_text"] for call in tts.calls]
    assert len(pieces) > 1
    assert all(len(piece) <= 500 for piece in pieces)
    # The pieces join with no gap: the audio is exactly the pieces' samples.
    samples, _rate = decode_wav(reply["audio"])
    assert samples.size == sum(len(piece) * 10 for piece in pieces)


def test_the_sample_rate_comes_from_the_model():
    reply = child(FakeTTS(sample_rate=44100.0)).speak("Hi.", "t", references={"t": ref(b"c")})
    assert reply["sample_rate"] == 44100
    assert decode_wav(reply["audio"])[1] == 44100


def test_reply_shape():
    reply = child().speak("One. Two.", "t", segments=["One.", "Two."],
                          references={"t": ref(b"c")})
    assert set(reply) == {"audio", "sample_rate", "model", "revision", "segments",
                          "conditioning", "adapters"}
    assert (reply["model"], reply["revision"]) == ("voxcpm2", REVISION)
    assert [s["text"] for s in reply["segments"]] == ["One.", "Two."]
    assert set(reply["conditioning"]["t"]) == {"state", "model", "revision", "reason", "blob"}
    assert reply["audio"][:4] == b"RIFF"


def _top_level_imports(node) -> list[str]:
    """Modules imported at import time — everything outside a function body."""
    names: list[str] = []
    for child_node in ast.iter_child_nodes(node):
        if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(child_node, ast.Import):
            names += [alias.name for alias in child_node.names]
        elif isinstance(child_node, ast.ImportFrom):
            names.append(child_node.module or "")
        else:
            names += _top_level_imports(child_node)
    return names


def test_the_module_imports_no_torch_or_voxcpm_at_module_scope():
    tree = ast.parse(Path(voxcpm_voice.__file__).read_text())
    roots = {name.split(".")[0] for name in _top_level_imports(tree)}
    assert not roots & {"torch", "torchaudio", "voxcpm", "huggingface_hub", "transformers"}


LORA = json.dumps({"enable_lm": True, "enable_dit": True, "enable_proj": False,
                   "r": 8, "alpha": 16, "dropout": 0.0,
                   "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
                   "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]},
                  sort_keys=True)
LORA_A = "base_lm.layers.0.self_attn.q_proj.lora_A"
LORA_B = "base_lm.layers.0.self_attn.q_proj.lora_B"
BASE = "base_lm.layers.0.self_attn.q_proj.weight"
A_SHAPE = (2, 4)
B_SHAPE = (4, 2)
METADATA = {
    "format": "ivms777-voxcpm2-lora-1", "model": "voxcpm2", "revision": REVISION,
    "voxcpm": VERSION, "lora": LORA, "dataset_sha256": "d" * 64, "config": "{}",
    "steps_done": "200",
}


class FakeLoRATTS(FakeTTS):
    """A `tts_model` built with LoRA layers, with the pinned package's load rules:
    `load_lora_weights` copies into ANY parameter of a matching name and skips the
    rest (`voxcpm2.py:1229-1241`); a reset zeroes B (`lora.py:67-71`)."""

    def __init__(self):
        super().__init__()
        self.params = {
            BASE: np.ones((4, 4), dtype=np.float32),
            LORA_A: np.full(A_SHAPE, 0.5, dtype=np.float32),
            LORA_B: np.zeros(B_SHAPE, dtype=np.float32),
        }
        self.events: list = []
        self.load_paths: list[tuple[str, bool]] = []
        self.fail_after_one = False
        self.extra_skipped: list[str] = []

    def get_lora_state_dict(self):
        return {name: value.copy() for name, value in self.params.items() if "lora_" in name}

    def reset_lora_weights(self):
        self.events.append("reset")
        self.params[LORA_B][...] = 0.0

    def load_lora_weights(self, path):
        from safetensors.numpy import load_file

        self.events.append("load")
        self.load_paths.append((path, os.path.exists(path)))
        loaded, skipped = [], []
        for key, value in load_file(path).items():
            if key not in self.params:
                skipped.append(key)
                continue
            np.copyto(self.params[key], value)
            loaded.append(key)
            if self.fail_after_one:
                raise RuntimeError("copy failed halfway")
        return loaded, skipped + self.extra_skipped

    def generate_with_prompt_cache(self, **kwargs):
        # The B value tells which adapter this generation ran with (0 = none).
        self.events.append(("generate", kwargs["target_text"], float(self.params[LORA_B][0, 0])))
        return super().generate_with_prompt_cache(**kwargs)


def adapter(b_value: float, *, metadata=None, tensors=None) -> dict:
    """A baked voice model as the child receives it: decoded bytes and their sha256."""
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    if tensors is None:
        tensors = {LORA_A: np.full(A_SHAPE, 0.5, dtype=np.float32),
                   LORA_B: np.full(B_SHAPE, b_value, dtype=np.float32)}
    blob = safetensors_numpy.save(tensors, metadata={**METADATA, **(metadata or {})})
    return {"blob": blob, "sha256": hashlib.sha256(blob).hexdigest()}


def baked(clip: bytes, model: dict | None, use_clip: bool = True) -> dict:
    return {**ref(clip), "adapter": model, "use_clip": use_clip}


def lora_child(tts: FakeLoRATTS) -> VoxCPMVoice:
    return VoxCPMVoice.loaded(tts, REVISION, save=pickle.dumps, load=pickle.loads,
                              version=VERSION, lora=LORA)


def operations(tts: FakeLoRATTS) -> list[str]:
    return [event for event in tts.events if isinstance(event, str)]


def generations(tts: FakeLoRATTS) -> list[tuple[str, float]]:
    return [(event[1], event[2]) for event in tts.events if isinstance(event, tuple)]


def test_an_adapter_is_loaded_kept_and_reset_per_request():
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    x = adapter(1.0)

    plain = voice.speak("Hi.", "t", references={"t": baked(b"c", None)})
    assert plain["adapters"] == {"t": {"state": "none", "sha256": None}}
    assert operations(tts) == []

    first = voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    assert first["adapters"] == {"t": {"state": "loaded", "sha256": x["sha256"]}}
    assert operations(tts) == ["reset", "load"]

    again = voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    assert again["adapters"]["t"]["state"] == "kept"
    assert operations(tts) == ["reset", "load"]

    back = voice.speak("Hi.", "t", references={"t": baked(b"c", None)})
    assert back["adapters"]["t"]["state"] == "none"
    assert operations(tts) == ["reset", "load", "reset"]
    assert [value for _text, value in generations(tts)] == [0.0, 1.0, 1.0, 0.0]


def test_two_adapters_switch_in_speaking_order():
    tts = FakeLoRATTS()
    x, y = adapter(1.0), adapter(2.0)
    reply = lora_child(tts).speak(
        "One. Two. Three. Four.", "a", segments=["One.", "Two.", "Three.", "Four."],
        voices=[None, "b", None, "b"],
        references={"a": baked(b"clip-a", x), "b": baked(b"clip-b", y)},
    )
    assert operations(tts) == ["reset", "load"] * 4
    assert generations(tts) == [("One.", 1.0), ("Two.", 2.0), ("Three.", 1.0), ("Four.", 2.0)]
    assert reply["adapters"] == {"a": {"state": "loaded", "sha256": x["sha256"]},
                                 "b": {"state": "loaded", "sha256": y["sha256"]}}


@pytest.mark.parametrize(("metadata", "reason"), [
    ({"revision": "1" * 40}, "made for another model revision"),
    ({"lora": json.dumps({"enable_lm": True, "r": 16})}, "made with another LoRA shape"),
    ({"lora": "not json"}, "made with another LoRA shape"),
    ({"format": "someone-elses-lora"}, "not a voice model of this app"),
    ({"model": "voxcpm1"}, "not a voice model of this app"),
])
def test_adapter_mismatch_names_its_reason(metadata, reason):
    assert adapter_mismatch({**METADATA, **metadata}, REVISION, LORA) == reason


def test_adapter_mismatch_accepts_the_same_lora_in_another_key_order():
    shuffled = json.dumps(dict(reversed(list(json.loads(LORA).items()))))
    assert adapter_mismatch({**METADATA, "lora": shuffled}, REVISION, LORA) is None
    assert adapter_mismatch({}, REVISION, LORA) == "not a voice model of this app"


def test_tensor_mismatch_wants_exactly_the_lora_names_and_shapes():
    expected = {LORA_A: A_SHAPE, LORA_B: B_SHAPE}
    assert tensor_mismatch(dict(expected), expected) is None
    assert tensor_mismatch({**expected, BASE: (4, 4)}, expected) == (
        "the voice model holds 1 keys that are not LoRA weights of this model")
    assert tensor_mismatch({LORA_A: A_SHAPE}, expected) == "the voice model misses 1 keys"
    assert tensor_mismatch({LORA_A: A_SHAPE, LORA_B: (2, 4)}, expected) == (
        "1 tensors of the voice model have another shape")


def test_load_mismatch_names_a_skipped_or_a_missing_key():
    expected = {LORA_A, LORA_B}
    assert load_mismatch([LORA_A, LORA_B], [], expected) is None
    assert load_mismatch([LORA_A, LORA_B], ["other.lora_A"], expected) == (
        "the load skipped 1 keys of the voice model")
    assert load_mismatch([LORA_A], [], expected) == "the voice model misses 1 keys"


def assert_it_recovers(voice: VoxCPMVoice, tts: FakeLoRATTS, previous: dict) -> None:
    """After a refused adapter: a plain request speaks with no adapter, and the adapter
    that was loaded before loads again from zero — never `kept`."""
    plain = voice.speak("Plain.", "t", references={"t": baked(b"c", None)})
    assert plain["adapters"]["t"]["state"] == "none"
    again = voice.speak("Again.", "t", references={"t": baked(b"c", previous)})
    assert again["adapters"]["t"]["state"] == "loaded"
    assert generations(tts)[-2:] == [("Plain.", 0.0), ("Again.", 1.0)]


@pytest.mark.parametrize(("metadata", "reason"), [
    ({"revision": "1" * 40}, "made for another model revision"),
    ({"lora": json.dumps({"r": 4})}, "made with another LoRA shape"),
    ({"format": "someone-elses-lora"}, "not a voice model of this app"),
])
def test_an_adapter_with_foreign_metadata_is_refused_before_any_load(metadata, reason):
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    x = adapter(1.0)
    voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    spoken = len(generations(tts))

    with pytest.raises(ValueError, match=reason):
        voice.speak("Bad.", "t", references={"t": baked(b"c", adapter(3.0, metadata=metadata))})
    assert operations(tts) == ["reset", "load", "reset"]
    assert len(generations(tts)) == spoken
    assert_it_recovers(voice, tts, x)


def test_a_base_weight_in_the_file_is_refused_and_never_copied():
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    x = adapter(1.0)
    voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    poisoned = adapter(3.0, tensors={
        LORA_A: np.full(A_SHAPE, 0.5, dtype=np.float32),
        LORA_B: np.full(B_SHAPE, 3.0, dtype=np.float32),
        BASE: np.zeros((4, 4), dtype=np.float32),
    })

    with pytest.raises(ValueError, match="1 keys that are not LoRA weights"):
        voice.speak("Bad.", "t", references={"t": baked(b"c", poisoned)})
    assert operations(tts) == ["reset", "load", "reset"]
    assert np.all(tts.params[BASE] == 1.0)
    assert_it_recovers(voice, tts, x)


def test_a_tensor_of_another_shape_is_refused_before_the_load():
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    x = adapter(1.0)
    voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    reshaped = adapter(3.0, tensors={LORA_A: np.full(A_SHAPE, 0.5, dtype=np.float32),
                                     LORA_B: np.full((8, 1), 3.0, dtype=np.float32)})

    with pytest.raises(ValueError, match="1 tensors of the voice model have another shape"):
        voice.speak("Bad.", "t", references={"t": baked(b"c", reshaped)})
    assert operations(tts) == ["reset", "load", "reset"]
    assert_it_recovers(voice, tts, x)


def test_a_load_that_fails_halfway_resets_and_forgets_the_kept_adapter():
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    x = adapter(1.0)
    voice.speak("Hi.", "t", references={"t": baked(b"c", x)})
    tts.fail_after_one = True

    with pytest.raises(ValueError, match="copy failed halfway"):
        voice.speak("Bad.", "t", references={"t": baked(b"c", adapter(3.0))})
    assert operations(tts) == ["reset", "load", "reset", "load", "reset"]
    assert tts.params[LORA_B][0, 0] == 0.0
    tts.fail_after_one = False
    assert_it_recovers(voice, tts, x)


def test_a_key_the_load_skipped_resets_and_nothing_is_spoken():
    tts = FakeLoRATTS()
    tts.extra_skipped = ["residual_lm.layers.0.self_attn.q_proj.lora_A"]

    with pytest.raises(ValueError, match="the load skipped 1 keys of the voice model"):
        lora_child(tts).speak("Hi.", "t", references={"t": baked(b"c", adapter(1.0))})
    assert operations(tts) == ["reset", "load", "reset"]
    assert generations(tts) == []


def test_without_an_adapter_the_lora_child_speaks_as_a_plain_child():
    lora_tts, plain_tts = FakeLoRATTS(), FakeTTS()
    references = {"a": baked(b"clip-a", None), "b": baked(b"clip-b", None)}
    request = {"segments": ["One.", "Two."], "voices": [None, "b"], "references": references}

    lora_child(lora_tts).speak("One. Two.", "a", **request)
    child(plain_tts).speak("One. Two.", "a", **request)
    assert operations(lora_tts) == []
    assert lora_tts.calls == plain_tts.calls


def test_use_clip_false_speaks_from_the_adapter_alone():
    tts = FakeLoRATTS()
    x = adapter(1.0)
    reply = lora_child(tts).speak("Hi.", "t", references={"t": baked(b"c", x, use_clip=False)})
    assert tts.calls[0]["prompt_cache"] is None
    assert tts.built == []
    assert reply["conditioning"] == {}
    assert reply["adapters"]["t"]["state"] == "loaded"


def test_a_model_only_reference_with_no_audio_speaks_from_the_adapter_alone():
    tts = FakeLoRATTS()
    x = adapter(1.0)
    reply = lora_child(tts).speak("Hi.", "t", references={
        "t": {"language": "ru", "adapter": x, "use_clip": False}})
    assert tts.built == []
    assert tts.calls[0]["prompt_cache"] is None
    assert reply["conditioning"] == {}
    assert reply["adapters"] == {"t": {"state": "loaded", "sha256": x["sha256"]}}


def test_an_adapter_to_a_child_built_without_lora_is_refused():
    tts = FakeTTS()
    with pytest.raises(ValueError, match="^this voice child was built without LoRA$"):
        child(tts).speak("Hi.", "t", references={"t": baked(b"c", adapter(1.0))})
    assert tts.calls == []


def test_the_temporary_adapter_file_is_a_safetensors_file_that_is_removed(monkeypatch):
    made: list[str] = []
    mkstemp = voxcpm_voice.tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        handle, path = mkstemp(*args, **kwargs)
        made.append(path)
        return handle, path

    monkeypatch.setattr(voxcpm_voice.tempfile, "mkstemp", recording_mkstemp)
    tts = FakeLoRATTS()
    voice = lora_child(tts)
    voice.speak("Hi.", "t", references={"t": baked(b"c", adapter(1.0))})
    with pytest.raises(ValueError):
        voice.speak("Bad.", "t", references={"t": baked(b"c", adapter(
            3.0, metadata={"revision": "1" * 40}))})
    tts.fail_after_one = True
    with pytest.raises(ValueError):
        voice.speak("Bad.", "t", references={"t": baked(b"c", adapter(4.0))})

    adapter_files = [path for path in made if not path.endswith(".wav")]
    assert len(adapter_files) == 3
    assert all(path.endswith(".safetensors") for path in adapter_files)
    assert not any(os.path.exists(path) for path in adapter_files)
    assert [existed for _path, existed in tts.load_paths] == [True, True]


def test_a_lora_child_is_built_with_the_catalog_lora_config(monkeypatch):
    snapshot_at(monkeypatch, REVISION)
    monkeypatch.setattr(voxcpm_voice, "_voxcpm_version", lambda: VERSION)
    monkeypatch.setattr(voxcpm_voice, "_torch_save", pickle.dumps)
    monkeypatch.setattr(voxcpm_voice, "_torch_load", pickle.loads)
    calls = []

    class LoRAConfig:
        def __init__(self, **fields):
            self.fields = fields

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            model = cls()
            model.tts_model = FakeLoRATTS()
            return model

    model_pkg = types.ModuleType("voxcpm.model")
    voxcpm2 = types.ModuleType("voxcpm.model.voxcpm2")
    voxcpm2.LoRAConfig = LoRAConfig
    model_pkg.voxcpm2 = voxcpm2
    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=FakeVoxCPM,
                                                                     model=model_pkg))
    monkeypatch.setitem(sys.modules, "voxcpm.model", model_pkg)
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm2", voxcpm2)

    voice = VoxCPMVoice("openbmb/VoxCPM2", "mps", REVISION, LORA)
    [kwargs] = calls
    assert {k: v for k, v in kwargs.items() if k != "lora_config"} == {
        "load_denoiser": False, "optimize": False, "device": "mps"}
    assert kwargs["lora_config"].fields == json.loads(LORA)
    reply = voice.speak("Hi.", "t", references={"t": baked(b"c", adapter(1.0))})
    assert reply["adapters"]["t"]["state"] == "loaded"
