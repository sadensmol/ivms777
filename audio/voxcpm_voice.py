"""VoxCPM2 voice copy in a `TorchWorker` child (design §5.1, §8.1, §11.1).

Models-service-only, like every other module in `audio/`. It speaks a SAVED voice: each
voice token in a request carries one example WAV. An example with a transcript is
cloned from the clip AND its words (`ref_continuation`), which keeps the person's pace;
one without is cloned from the clip alone (`reference`), which keeps only the timbre.

Module scope imports no torch, no voxcpm, no huggingface_hub and no safetensors, exactly
like `audio/silero_voice.py`: the pure helpers (`check_references`, `pick_conditioning`,
`payload_mismatch`, `adapter_mismatch`, `tensor_mismatch`, `load_mismatch`) are
unit-tested, and importing them must not build a torch context in the importing process
(CLAUDE.md § "One model process").

Building the conditioning encodes the clip with the AudioVAE, so the child returns it
as an opaque blob. A film or test that sends a stored blob back has the child restore
it when its model key, revision, voxcpm version and clip sha256 match.
voxcpm 2.0.3 has no save/load API for a prompt cache, so the blob is a `torch.save`
of a small payload dict.

A reference may also carry a baked voice model: a LoRA adapter a bake wrote (§4.1,
§5.1). A child built with the catalog's LoRA shape keeps ONE adapter loaded, by sha256,
and puts the right one in place before each run of a token.

A reference may set `controls` for a test run (§5.1): they cover `CONTROLS` for that token
only, and the reply echoes them. No film sets them.

Licence: Apache-2.0 for the code and the weights.
"""

import json
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np

from audio.segments import speak_passages
from audio.silero_voice import split_text
from storage.wav import encode_wav

MODEL_KEY = "voxcpm2"
PAYLOAD_FORMAT = 1
ADAPTER_FORMAT = "ivms777-voxcpm2-lora-1"
# The package's own mode names (`voxcpm2.py:731-738`).
MODES = ("reference", "ref_continuation")
# The package's own `generate` defaults (`core.py:180-195`). 2.0.3 has no seed.
CONTROLS = {
    "min_len": 2,
    "max_len": 4096,
    "inference_timesteps": 10,
    "cfg_value": 2.0,
    "retry_badcase": True,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0,
}


def check_references(voice: str | None, voices: list[str | None] | None,
                     references: dict | None) -> list[str]:
    """The voice tokens this request speaks, in first-use order.

    `voice` and every non-empty `voices[i]` must be keys of `references`: this child has
    no preset speakers, so a name without an example clip cannot be spoken.
    """
    if not references:
        raise ValueError("VoxCPM2 speaks saved voices only: the request has no references")
    used: list[str] = []
    for token in [voice, *(name for name in (voices or []) if name)]:
        if token not in references:
            raise ValueError(f"voice {token!r} has no reference clip")
        if token not in used:
            used.append(token)
    return used


def pick_conditioning(ref: dict, model: str, revision: str) -> tuple[bytes | None, str | None]:
    """`(blob, None)` for the stored conditioning of `model` at `revision`, else
    `(None, reason)`.

    The newest match wins: the app appends a blob each time the child builds one.
    """
    stored = [c for c in (ref.get("conditioning") or []) if c.get("model") == model]
    if not stored:
        return None, "no stored conditioning"
    matching = [c for c in stored if c.get("revision") == revision]
    if not matching:
        return None, "stored for another revision"
    return matching[-1]["blob"], None


def _transcript(ref: dict) -> str | None:
    """The example's own words, or `None` when it has none (an uploaded clip)."""
    text = ref.get("transcript")
    return text if isinstance(text, str) and text.strip() else None


def _mode(transcript: str | None) -> str:
    """Reference mode copies the TIMBRE only, and the pace is the model's own: a Russian
    film came out a third faster than the person's clip. With the clip's words the model
    continues from their own speech — the package's "Ultimate Cloning", which keeps
    "timbre, rhythm, emotion, and style"."""
    return "ref_continuation" if transcript else "reference"


def payload_mismatch(payload, revision: str, version: str, sha256: str,
                     transcript: str | None) -> str | None:
    """Why a restored payload cannot be used, or `None` when every field matches."""
    if not isinstance(payload, dict) or not isinstance(payload.get("cache"), dict):
        return "stored conditioning is invalid"
    if (payload.get("format") != PAYLOAD_FORMAT or payload.get("model") != MODEL_KEY
            or payload.get("mode") not in MODES):
        return "stored conditioning is invalid"
    if payload.get("revision") != revision:
        return "stored for another revision"
    if payload.get("voxcpm") != version:
        return "voxcpm version changed"
    if payload.get("sha256") != sha256:
        return "clip changed"
    if payload.get("mode") != _mode(transcript):
        return "cloning mode changed"
    if payload.get("transcript") != transcript:
        return "transcript changed"
    return None


def adapter_mismatch(metadata: dict, revision: str, lora: str) -> str | None:
    """Why an adapter file's metadata does not fit this child, or `None`."""
    if metadata.get("format") != ADAPTER_FORMAT or metadata.get("model") != MODEL_KEY:
        return "not a voice model of this app"
    if metadata.get("revision") != revision:
        return "made for another model revision"
    try:
        same_shape = json.loads(metadata.get("lora") or "null") == json.loads(lora)
    except json.JSONDecodeError:
        same_shape = False
    return None if same_shape else "made with another LoRA shape"


def tensor_mismatch(shapes: dict[str, tuple], expected: dict[str, tuple]) -> str | None:
    """Why the file's tensors are not exactly this model's LoRA weights, or `None`.

    Checked BEFORE the load: `load_lora_weights` copies into ANY parameter of a matching
    name, base weights included (`voxcpm2.py:1229-1241`), and a reset restores only LoRA.
    """
    extra = shapes.keys() - expected.keys()
    if extra:
        return f"the voice model holds {len(extra)} keys that are not LoRA weights of this model"
    missing = expected.keys() - shapes.keys()
    if missing:
        return f"the voice model misses {len(missing)} keys"
    other = [name for name, shape in shapes.items() if tuple(shape) != tuple(expected[name])]
    if other:
        return f"{len(other)} tensors of the voice model have another shape"
    return None


def load_mismatch(loaded: list, skipped: list, expected: set) -> str | None:
    """Why a finished load did not set every LoRA weight from the file, or `None`."""
    if skipped:
        return f"the load skipped {len(skipped)} keys of the voice model"
    missing = expected - set(loaded)
    if missing:
        return f"the voice model misses {len(missing)} keys"
    return None


def _normalise(text: str) -> str:
    # The same whitespace clean-up as the package's own `generate` (`core.py:262-263`).
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _voxcpm_version() -> str:
    from importlib.metadata import version

    return version("voxcpm")


def _torch_save(payload: dict) -> bytes:
    import torch

    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _torch_load(blob: bytes):
    import torch

    return torch.load(BytesIO(blob), map_location="cpu", weights_only=True)


class VoxCPMVoice:
    def __init__(self, repo: str, device: str, revision: str, lora: str | None = None) -> None:
        # GPU only (`docs/design.md` §5): voxcpm raises on an absent mps/cuda itself, but
        # "cpu" or None would silently run a 2.3B model on the CPU.
        if device != "mps" and not str(device).startswith("cuda"):
            raise RuntimeError(f"VoxCPM2 runs on mps or cuda only, not {device!r}")

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # A Hub id would make voxcpm download `main` (`core.py:157-161`), so the child
        # resolves the PINNED snapshot itself, and never downloads inside a film.
        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"VoxCPM2 weights for {revision} are not on disk: run `make models`, "
                "or POST /models/download {slot: voice, key: voxcpm2}"
            ) from exc
        # The snapshot folder is the commit; anything else is another revision.
        if Path(path).name != revision:
            raise RuntimeError(
                f"VoxCPM2 snapshot {Path(path).name!r} is not the pinned revision {revision!r}"
            )

        from voxcpm import VoxCPM

        # No denoiser (ModelScope) and no torch.compile warm-up (`core.py:89-101`). MPS
        # runs in float32 because the package forces it; no env override is set here.
        if lora is None:
            model = VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False,
                                           device=device)
        else:
            # LoRA layers exist only when the model is BUILT with a config
            # (`voxcpm2.py:246-247`). With no adapter loaded every B is zero.
            from voxcpm.model.voxcpm2 import LoRAConfig

            model = VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False,
                                           device=device,
                                           lora_config=LoRAConfig(**json.loads(lora)))
        self.repo = repo
        self.device = device
        self._setup(model.tts_model, revision, _torch_save, _torch_load, _voxcpm_version(), lora)

    @classmethod
    def loaded(cls, tts, revision: str, *, save, load, version: str = "test",
               lora: str | None = None) -> "VoxCPMVoice":
        """A child around an already built `tts_model` — the test seam.

        `save` and `load` replace `torch.save` / `torch.load`, so a test drives every
        path with pickle and a fake model, and never imports torch.
        """
        voice = cls.__new__(cls)
        voice.repo = None
        voice.device = None
        voice._setup(tts, revision, save, load, version, lora)
        return voice

    def _setup(self, tts, revision: str, save, load, version: str, lora: str | None) -> None:
        self._tts = tts
        self.revision = revision
        self.sample_rate = int(tts.sample_rate)
        self._save = save
        self._load = load
        self._version = version
        self._lora = lora
        self._lora_shapes = None if lora is None else {
            name: tuple(value.shape) for name, value in tts.get_lora_state_dict().items()
        }
        self._adapter_sha: str | None = None

    def warm(self) -> None:
        """Resident must MEAN resident (§8.1): one short zero-shot call at load time."""
        self._tts.generate_with_prompt_cache(
            target_text="Ready.", prompt_cache=None, max_len=64, retry_badcase=False
        )

    def speak(self, text: str, voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None,
              references: dict | None = None) -> dict:
        """One WAV for the whole script, every passage in its saved voice.

        Same-voice passages are one run (`audio/segments.py`). A run is cut again into
        sentence pieces of at most 500 characters, because one long autoregressive call
        has no measured behaviour on VoxCPM2; the pieces join with no gap.
        """
        used = check_references(voice, voices, references)
        caches: dict[str, dict | None] = {}
        conditioning: dict[str, dict] = {}
        adapters: dict[str, dict] = {}
        echoed: dict[str, dict] = {}
        for token in used:
            if references[token].get("use_clip", True) is False:
                caches[token] = None
                continue
            cache, state, blob, reason = self._conditioning(token, references[token])
            caches[token] = cache
            conditioning[token] = {
                "state": state, "model": MODEL_KEY, "revision": self.revision,
                "reason": reason, "blob": blob,
            }

        def speak_one(passage: str, token: str):
            ref = references[token]
            adapters[token] = {"state": self._use_adapter(ref),
                               "sha256": (ref.get("adapter") or {}).get("sha256")}
            controls = {**CONTROLS, **ref.get("controls", {})}
            if "controls" in ref:
                echoed[token] = {name: controls[name] for name in ref["controls"]}
            spoken = []
            for piece in split_text(_normalise(passage)):
                wav, _, _ = self._tts.generate_with_prompt_cache(
                    target_text=piece, prompt_cache=caches[token], **controls
                )
                spoken.append(np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32))
            return np.concatenate(spoken) if spoken else np.zeros(1, dtype=np.float32)

        chunks = [c for c in (segments or []) if c and c.strip()] or [text]
        samples, timings = speak_passages(
            chunks, list(voices or []), voice, speak_one, self.sample_rate
        )
        reply = {
            "audio": encode_wav(samples, self.sample_rate),
            "sample_rate": self.sample_rate,
            "model": MODEL_KEY,
            "revision": self.revision,
            "segments": timings,
            "conditioning": conditioning,
            "adapters": adapters,
        }
        return reply | {"controls": echoed} if echoed else reply

    def _use_adapter(self, ref: dict) -> str:
        """Put the token's adapter in place before its run: `none`, `kept` or `loaded`.

        Any failure resets the LoRA weights and forgets the kept sha256, so no later run
        reads `kept` for a half-loaded model, and the next adapter loads from zero.
        """
        adapter = ref.get("adapter")
        if adapter is None:
            if self._adapter_sha is not None:
                self._tts.reset_lora_weights()
                self._adapter_sha = None
            return "none"
        if self._lora_shapes is None:
            raise ValueError("this voice child was built without LoRA")
        if adapter["sha256"] == self._adapter_sha:
            return "kept"
        try:
            reason = self._load_adapter(adapter["blob"])
        except Exception as exc:
            self._tts.reset_lora_weights()
            self._adapter_sha = None
            raise ValueError(f"the voice model could not be loaded: {exc}") from exc
        if reason is not None:
            self._tts.reset_lora_weights()
            self._adapter_sha = None
            raise ValueError(reason)
        self._adapter_sha = adapter["sha256"]
        return "loaded"

    def _load_adapter(self, blob: bytes) -> str | None:
        """Check the adapter file, then reset and load it. The reason it was refused, or
        `None` once it is loaded."""
        # `load_lora_weights` picks its reader by the suffix (`voxcpm2.py:1216-1218`).
        handle, path = tempfile.mkstemp(suffix=".safetensors")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(blob)
            from safetensors import safe_open

            with safe_open(path, framework="np") as file:
                metadata = file.metadata() or {}
                shapes = {name: tuple(file.get_slice(name).get_shape())
                          for name in file.keys()}  # noqa: SIM118 - safe_open is not iterable
            reason = (adapter_mismatch(metadata, self.revision, self._lora)
                      or tensor_mismatch(shapes, self._lora_shapes))
            if reason is not None:
                return reason
            self._tts.reset_lora_weights()
            loaded, skipped = self._tts.load_lora_weights(path)
            return load_mismatch(loaded, skipped, set(self._lora_shapes))
        finally:
            os.unlink(path)

    def _conditioning(self, token: str, ref: dict) -> tuple[dict, str, bytes | None, str | None]:
        """`(cache, state, blob, reason)` for one voice token.

        `restored` reuses the stored blob (no blob goes back). `built` encodes the clip
        and returns the new blob with the reason the stored one was not used.
        """
        transcript = _transcript(ref)
        blob, reason = pick_conditioning(ref, MODEL_KEY, self.revision)
        if blob is not None:
            try:
                payload = self._load(blob)
            except Exception:  # noqa: BLE001 - an unreadable blob is rebuilt, not fatal
                payload = None
            reason = payload_mismatch(payload, self.revision, self._version, ref["sha256"],
                                      transcript)
            if reason is None:
                return payload["cache"], "restored", None, None

        handle, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(ref["audio"])
            if transcript:
                # The same clip as reference AND prompt, as the package's own example
                # does "for maximum cloning similarity"; the transcript as stored.
                cache = self._tts.build_prompt_cache(
                    prompt_text=transcript, prompt_wav_path=path, reference_wav_path=path)
            else:
                cache = self._tts.build_prompt_cache(reference_wav_path=path)
        finally:
            os.unlink(path)
        payload = {
            "format": PAYLOAD_FORMAT, "model": MODEL_KEY, "revision": self.revision,
            "voxcpm": self._version, "sha256": ref["sha256"], "mode": _mode(transcript),
            "transcript": transcript, "cache": cache,
        }
        return cache, "built", self._save(payload), reason
