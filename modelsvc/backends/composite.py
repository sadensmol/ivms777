"""`CompositeBackend` — a `ModelBackend` assembled from independent
sub-backends, one per concern (design §5.1), and the single funnel through
which every model op passes the conveyor (plan 18).

Each op acquires a scheduler slot (concurrency + priority) and asks the governor
to make the models it needs resident, then runs. Interactive ops (search/chat)
use `Priority.INTERACTIVE`; batch ingest captioning uses `Priority.BATCH`, so a
chat preempts a queued caption on the single-slot Jetson. When no `scheduler` is
injected (bare unit tests), ops run directly — same behaviour as before plan 18.

A sub-backend not yet wired raises `NotImplementedError` so a premature call
fails loudly.
"""

import logging
from collections.abc import Iterator
from contextlib import nullcontext

from models import catalog
from modelsvc.activity import Activity
from modelsvc.backends.base import (
    CloningUnavailable,
    SlotOff,
    SpeakerEmbedNotOffered,
    SpeakerEmbedRefused,
    TranscribeNotOffered,
    TranscribeRefused,
)
from modelsvc.scheduler import Priority
from modelsvc.slots import ASR_UNIT, LLM_UNITS, SPEAKER_EMBED_UNIT, TRAIN_UNIT, catalog_payload
from modelsvc.voice_bakes import BakeNotOffered, BakeOnly, VoiceBakes, VoiceGate

log = logging.getLogger(__name__)


class CompositeBackend:
    def __init__(
        self,
        *,
        embed,
        caption=None,
        text=None,
        voice=None,
        music=None,
        registry=None,
        governor=None,
        scheduler=None,
        text_embed_needs: tuple[str, ...] = ("text_embed",),
        slots=None,
        downloads=None,
        profile: str = "mac",
        activity: Activity | None = None,
        bakes: VoiceBakes | None = None,
    ) -> None:
        self._embed = embed
        self._caption = caption
        self._text = text
        self._voice = voice
        self._music = music
        self._registry = registry
        self._governor = governor
        self._scheduler = scheduler
        self._text_embed_needs = text_embed_needs
        # The `SlotManager` (design §4.1) — which model each slot holds, and the
        # generation `app` compares against. None in bare unit tests.
        self._slots = slots
        self._downloads = downloads
        self._profile = profile
        # The current in-flight op label, for the resource bar (§13). Shared with the
        # bakes, so a bake shows on the bar too.
        self._activity = activity if activity is not None else Activity()
        self._bakes = bakes
        self._gate = bakes.gate if bakes is not None else VoiceGate()

    # --- conveyor plumbing ---
    def _run(self, needed, priority, label, fn):
        def tracked():
            with self._activity.track(label):
                return fn()

        if self._scheduler is None:
            return tracked()
        result = self._scheduler.run(list(needed), priority, tracked)
        # An embedder op just ran; proactively unload the llm (either mode) if it has
        # gone idle past its TTL, freeing GB for the rest of a long batch (no-op on
        # mac/cloud).
        if not any(n.startswith("llm") for n in needed):
            self._scheduler.reap_idle(list(LLM_UNITS))
        return result

    # --- SigLIP ---
    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.embed_image(images))

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.embed_text(texts))

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "tagging",
                         lambda: self._embed.tag(image, dimensions))

    def calibration(self) -> dict:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.calibration())

    # --- captioning (llm_vision, batch) ---
    def caption(self, image: bytes) -> dict:
        # The ONLY op that sends an image to the llm, so the ONLY one that needs the
        # vision projector loaded (design §3.1). Chat/planner use text-only `llm`.
        if self._caption is None:
            raise NotImplementedError("caption backend not wired")
        return self._run(["llm_vision"], Priority.BATCH, "captioning",
                         lambda: self._caption.caption(image))

    # --- feed audio (voice / music units, batch) ---
    def _unit_available(self, unit: str) -> bool:
        # `SlotManager` registers NO residency unit for an `off` entry (§4.1), so
        # asking the scheduler for it would KeyError deep in the governor. Check
        # first and answer with the honest reason instead.
        return self._slots is None or self._slots.has_unit(unit)

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None,
              references: dict | None = None) -> dict:
        if self._voice is None:
            raise SlotOff("voice")
        # Counted in BEFORE routing: a speak that then waits for a scheduler slot is
        # still one the bake drains before it unloads `voice` (§8.1).
        with self._gate.admitted():
            if references is not None:
                return self._speak_saved(text, language, voice, segments, voices, references)
            return self._speak_preset(text, language, voice, segments, voices)

    def _speak_preset(self, text, language, voice, segments, voices) -> dict:
        # The LANGUAGE picks the model, not the user: a film asks to be narrated in
        # Russian and the manager swaps the voice unit to one that speaks Russian.
        # Only `off`, or no voice covering the language at all, is a refusal.
        if self._slots is not None:
            if not self._slots.ensure_voice_for(language):
                raise SlotOff(f"no voice available for {language}")
        elif not self._unit_available("voice"):
            raise SlotOff("voice")
        return self._run(["voice"], Priority.BATCH, "narrating",
                         lambda: self._voice.speak(text, voice, segments, voices))

    def _speak_saved(self, text, language, voice, segments, voices, references) -> dict:
        """Speak SAVED voices (§11.1): route to the cloning entry for the language, and
        check that the reply came from the model that was routed (D26). A stored voice
        identity must never be spoken by another model under its name."""
        def call() -> dict:
            return self._voice.speak(text, voice, segments, voices, references)

        if self._slots is None:  # bare unit tests: no routing, no identity to check
            return self._run(["voice"], Priority.BATCH, "narrating", call)
        if not self._slots.ensure_voice_for(language, cloning=True):
            raise CloningUnavailable(f"no voice-copy model speaks {language} on {self._profile}")
        entry = self._slots.entry("voice")
        result = self._run(["voice"], Priority.BATCH, "narrating", call)
        revision = getattr(entry.source, "revision", None)
        if result.get("model") != entry.key or result.get("revision") != revision:
            raise RuntimeError(
                f"voice identity mismatch: routed {entry.key}@{revision}, "
                f"the voice answered {result.get('model')}@{result.get('revision')}"
            )
        return result

    def transcribe(self, audio: bytes, language: str) -> dict:
        entry = catalog.transcriber_for(self._profile)
        if self._slots is None or not self._slots.has_unit(ASR_UNIT) or entry is None:
            raise TranscribeNotOffered("hearing the words of a take is not offered on this device yet")
        if language not in entry.languages:
            raise TranscribeRefused(f"{entry.key} does not hear {language!r}; "
                                    f"it hears {', '.join(entry.languages)}")
        revision = entry.source.revision
        # Counted in at the voice gate, so a bake's drain waits for this call and no
        # Whisper load starts beside the training child (§8.1).
        with self._gate.admitted():
            loaded_before = self._registry is not None and self._registry.is_resident(ASR_UNIT)
            before = self._governor.state() if self._governor is not None else None
            result = self._run([ASR_UNIT], Priority.BATCH, "hearing the words of a take",
                               lambda: self._slots.worker(ASR_UNIT).call("transcribe", audio,
                                                                         language))
            if (not loaded_before and before is not None and self._registry is not None
                    and self._registry.is_resident(ASR_UNIT)):
                self._log_asr_load(entry.key, revision, before, self._governor.state())
        if result.get("revision") != revision:
            raise RuntimeError(
                f"asr identity mismatch: routed {entry.key}@{revision}, "
                f"the child answered {result.get('revision')}"
            )
        return {"text": result["text"], "model": entry.key, "revision": revision,
                "seconds": result["seconds"], "elapsed_s": result["elapsed_s"]}

    def embed_voice(self, audio: bytes) -> dict:
        entry = catalog.speaker_embedder_for(self._profile)
        if entry is None or self._slots is None or not self._slots.has_unit(SPEAKER_EMBED_UNIT):
            raise SpeakerEmbedNotOffered("measuring a voice is not offered on this device yet")
        revision = entry.source.revision
        # Counted in at the voice gate, so a bake's drain waits for this call and no
        # load starts beside the training child (§8.1).
        with self._gate.admitted():
            result = self._run([SPEAKER_EMBED_UNIT], Priority.BATCH, "measuring a voice",
                               lambda: self._slots.worker(SPEAKER_EMBED_UNIT).call("embed", audio))
        if result.get("revision") != revision:
            raise RuntimeError(
                f"speaker_embed identity mismatch: routed {entry.key}@{revision}, "
                f"the child answered {result.get('revision')}"
            )
        if "refused" in result:
            raise SpeakerEmbedRefused(result["refused"])
        if len(result["vector"]) != entry.dim:
            raise RuntimeError(f"{entry.key} answered {len(result['vector'])} values, "
                               f"not {entry.dim}")
        return {"vector": result["vector"], "model": entry.key, "revision": revision,
                "seconds": result["seconds"], "elapsed_s": result["elapsed_s"],
                "pipeline": result["pipeline"]}

    def detect_faces(self, image: bytes) -> dict:
        return {"faces": [], "model": "", "revision": "", "elapsed_s": 0.0}

    @staticmethod
    def _log_asr_load(key: str, revision: str, before, after) -> None:
        # The net free-memory change of THIS request: the governor may have evicted
        # other units to make room, so it is never the model's own cost (§8.1).
        evicted = sorted(set(before.resident) - set(after.resident))
        log.info(
            "asr %s@%s loaded by this request: free_mb %.0f → %.0f, net change %.0f MB; evicted: %s",
            key, revision[:7], before.free_mb, after.free_mb, after.free_mb - before.free_mb,
            ", ".join(evicted) or "none",
        )

    def music(
        self, prompt: str = "", seconds: float = 30.0, seed: int | None = None,
        sections: list[dict] | None = None, temperature: float | None = None,
        guidance: float | None = None,
    ) -> dict:
        if self._music is None or not self._unit_available("music"):
            raise SlotOff("music")
        return self._run(["music"], Priority.BATCH, "composing",
                         lambda: self._music.music(prompt, seconds, seed,
                                                   sections=sections,
                                                   temperature=temperature,
                                                   guidance=guidance))

    # --- text generation (llm) ---
    def text_complete(
        self,
        model: str,
        messages: list[dict],
        json_schema: dict | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        return self._run(["llm"], Priority.INTERACTIVE, "planning",
                         lambda: self._text.text_complete(
                             model, messages, json_schema=json_schema,
                             temperature=temperature, max_tokens=max_tokens))

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        if self._text is None:
            raise NotImplementedError("text backend not wired")

        # A generator: hold the slot (and the llm) for the WHOLE stream, not just
        # until the generator object is built, so it is not evicted mid-answer.
        def stream() -> Iterator[str]:
            if self._scheduler is None:
                with self._activity.track("chat"):
                    yield from self._text.text_stream(model, messages)
                return
            with self._scheduler.reserve(["llm"], Priority.INTERACTIVE), self._activity.track("chat"):
                yield from self._text.text_stream(model, messages)

        return stream()

    def text_context(self) -> int | None:
        # Metadata, not generation: it reads the server's `/props` and holds no slot,
        # so it never queues behind a film being written.
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        return self._text.text_context()

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        return self._run(list(self._text_embed_needs), Priority.INTERACTIVE, "embedding",
                         lambda: self._text.text_embed(model, texts))

    def text_warm(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        self._text.text_warm(model)

    def text_evict(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        self._text.text_evict(model)

    # --- slot control (plan 21, design §4.1) ---
    def _slot_keys(self) -> dict:
        if self._slots is not None:
            return self._slots.state()["slots"]
        return dict(catalog.DEFAULTS[self._profile])

    def _generation(self) -> int:
        return self._slots.generation if self._slots is not None else 0

    def _download_status(self, entry) -> dict:
        if self._downloads is None:
            return {"state": "ready", "bytes": 0, "total": 0, "error": None}
        return self._downloads.status(entry)

    def catalog(self) -> dict:
        return catalog_payload(
            self._profile, self._slot_keys(), self._generation(), self._download_status
        )

    def set_slots(self, slots: dict) -> dict:
        if self._slots is None:
            raise ValueError("this backend has no switchable slots")
        self._slots.apply(slots)  # ValueError on an unknown / not-offered key
        return self._slots.state()

    def download(self, slot: str, key: str) -> dict:
        entry = catalog.get(slot, key)  # KeyError on an unknown model
        if self._downloads is None:
            raise ValueError("this backend downloads nothing")
        self._downloads.start(entry)
        return self._downloads.status(entry)

    def embed_spec(self) -> dict:
        """Calibration AND the image embedder's preprocessing contract, in one
        round trip (design §5.1). The caller resizes by this, never by a constant."""
        entry = (
            self._slots.entry("image_embed")
            if self._slots is not None
            else catalog.get("image_embed", catalog.default_key("image_embed", self._profile))
        )
        pre = entry.preprocess
        return self.calibration() | {
            "preprocess": {
                "input_px": pre.input_px,
                "resample": pre.resample,
                "mode": pre.mode,
            },
            "generation": self._generation(),
        }

    # --- conveyor control (plan 18, control API) ---
    def models_state(self) -> dict:
        slots = {"slots": self._slot_keys(), "generation": self._generation()}
        if self._governor is None:
            resident = list(self._registry.resident()) if self._registry is not None else []
            return {"resident": resident, "budget_mb": 0, "free_mb": 0.0,
                    "used_mb": 0, "active": self._activity.current()} | slots
        st = self._governor.state()
        return {"resident": st.resident, "budget_mb": st.budget_mb, "free_mb": st.free_mb,
                "used_mb": st.used_mb, "active": self._activity.current()} | slots

    def model_ensure(self, name: str) -> None:
        if name == TRAIN_UNIT:
            raise BakeOnly(f"{TRAIN_UNIT} is loaded only by a bake")
        # A manual load of a gated unit is counted in like a speak, so a bake drains it too.
        gated = name in ("voice", ASR_UNIT, SPEAKER_EMBED_UNIT)
        with self._gate.admitted() if gated else nullcontext():
            if self._scheduler is not None:
                self._scheduler.run([name], Priority.INTERACTIVE, lambda: None)
            elif self._registry is not None:
                self._registry.ensure(name)

    def model_unload(self, name: str) -> None:
        # Through the governor, which refuses a leased unit (`UnitLeased`).
        if self._governor is not None:
            self._governor.unload(name)
        elif self._registry is not None:
            self._registry.unload(name)

    def bake_start(self, request: dict) -> dict:
        if self._bakes is None:
            raise BakeNotOffered("baking a voice is not offered on this device yet")
        return self._bakes.start(request)

    def bake_status(self, bake_id: str) -> dict | None:
        return None if self._bakes is None else self._bakes.status(bake_id)

    def bake_current(self) -> dict | None:
        return None if self._bakes is None else self._bakes.current()

    def bake_cancel(self, bake_id: str) -> dict | None:
        return None if self._bakes is None else self._bakes.cancel(bake_id)

    def bake_adapter(self, bake_id: str) -> bytes | None:
        return None if self._bakes is None else self._bakes.adapter(bake_id)

    def bake_delete(self, bake_id: str) -> bool:
        return self._bakes is not None and self._bakes.delete(bake_id)

    def resources(self) -> dict:
        """What this service ALONE knows: which models are resident, and the op
        in flight. No machine metrics — RAM/CPU/GPU-load/temperature are host-wide
        sysfs/psutil reads that need no CUDA context, so `app` reads them itself
        and the bar keeps working when this service is down (design §5.1, §13).
        """
        active = self._activity.current()
        if self._registry is not None:
            resident = list(self._registry.resident())
        else:
            resident = ["image_embed"] if self._embed is not None else []
        # The text model is reported by the registry now (it is a managed
        # model). When no registry is wired (bare tests), fall back to the text
        # backend's own view so the resource bar still shows it.
        if self._registry is None and self._text is not None:
            text_fn = getattr(self._text, "resident_models", None)
            if text_fn is not None:
                for model in text_fn():
                    if model not in resident:
                        resident.append(model)
        return {
            "resident": resident,
            "active": active,
            "slots": self._slot_keys(),
            "generation": self._generation(),
        }
