"""The model catalog — the closed list of models each slot may hold (design §4.1).

Plain data. No torch, no transformers, no HTTP: `app`, `worker` and the `models`
service all import this, and the one-model-process rule (CLAUDE.md, §5.1) means it
must stay importable from a process that owns no model.

An entry carries everything its consumers need so that nothing downstream has to
special-case a model name:

- the `models` service needs `source` (what to spawn / load) and `cost_mb` (what the
  governor budgets against, §8.1);
- `RemoteEmbedder` needs `preprocess` (how to resize before the wire, §5.1);
- `app` needs `dim` (does switching rebuild `photo_vec`?), `size_mb` and
  `cost_measured` (what the settings popup shows, §13).

`cost_mb` is the JETSON figure — the binding profile. It is `cost_measured=True`
only when it was measured on the board the way §8.1 requires: the drop in
`psutil.virtual_memory().available` across a load. Everything else is an estimate
and is labelled as one in the UI, because an under-estimate OOMs the board and an
over-estimate makes the model unloadable.
"""

from __future__ import annotations

from dataclasses import dataclass

SLOTS: tuple[str, ...] = (
    "image_embed", "text_embed", "caption", "planner", "voice", "music",
)

# Which ingest stages a slot switch invalidates, inclusive (design §4.1). `None`
# means the slot stores nothing, so switching it re-runs nothing.
INVALIDATES: dict[str, tuple[str, str] | None] = {
    "image_embed": ("embed", "taxonomy"),
    "text_embed": ("caption_embed", "caption_embed"),
    "caption": ("caption", "caption_embed"),
    "planner": None,
    # The feed's models store nothing in the ingest pipeline (§11.1), so switching
    # either re-runs no stage. A prepared presentation is invalidated by the MEMORY
    # changing (§6's cascade), never by the voice changing — a switch is free.
    "voice": None,
    "music": None,
}


@dataclass(frozen=True)
class Preprocess:
    """How the CALLER must resize an image before sending it (design §5.1).

    `squash` stretches to `input_px`×`input_px` ignoring aspect ratio — what a
    fixed-resolution SigLIP processor does to any input it is given, so doing it
    client-side is an identity op that saves a full-resolution PNG encode.
    `native` fits the image inside `input_px` with the aspect ratio kept, for
    NaFlex / native-resolution encoders where squashing would destroy signal.
    """

    input_px: int
    resample: str  # "bilinear" | "bicubic"
    mode: str  # "squash" | "native"


@dataclass(frozen=True)
class HfSource:
    """A `transformers` checkpoint, fetched into the HF cache.

    `revision` pins one Hub commit. `None` follows the repo's `main`, as every entry
    did before pinning existed.
    """

    repo: str
    revision: str | None = None


@dataclass(frozen=True)
class GgufSource:
    """A llama.cpp GGUF (+ optional vision projector), fetched into `data_dir/models`."""

    repo: str
    file: str
    mmproj_repo: str | None = None
    mmproj_file: str | None = None


@dataclass(frozen=True)
class UrlSource:
    """A single file fetched over plain HTTP into `data_dir/models`.

    For weights that are NOT on the Hub at all. Silero publishes its TTS as one
    `torch.package` archive on its own host — there is no `silero/*` HF repo — so an
    `HfSource` would make the downloader probe a repo that does not exist and report
    the model as permanently absent. The GGUF path already streams files this way, so
    `Downloads` reuses that fetcher.
    """

    url: str
    file: str


@dataclass(frozen=True)
class Speaker:
    """One voice a `voice` model can read in, with what it SOUNDS like.

    This is metadata about the model, so it belongs beside the model — not hardcoded in
    whatever happens to pick a narrator. `feed/voice.py` reads it, the story pass is
    shown it and casts from it, and code validates the choice against this list: a name
    that is not here would make the voice model raise, and narration is mandatory
    (§11.1), so a bad cast would fail the whole film.

    Every name here has been verified to load and produce audio (CLAUDE.md § "Verify a
    model RUNS").
    """

    name: str        # exactly what `POST /speak` takes
    gender: str      # "female" | "male"
    register: str    # "bright" | "warm" | "neutral" | "deep"
    note: str        # what it sounds like, for whoever is casting


@dataclass(frozen=True)
class AdapterSpec:
    """A cloning entry that can learn ONE saved voice as a LoRA adapter (design §4.1, §8.1)."""

    lora: dict  # LoRAConfig fields, the same for the trainer and the voice child
    train_cost_mb: int  # resident cost of the training child — an ESTIMATE until a bake measures it
    train_cost_measured: bool
    train_profiles: tuple[str, ...]  # where a bake is offered; adapters are USED wherever the entry runs
    guidance_minutes: tuple[float, float]  # the package authors' figure: guidance, never a promise


@dataclass(frozen=True)
class ModelEntry:
    key: str
    slot: str
    display: str
    source: HfSource | GgufSource | UrlSource
    size_mb: int  # download size, for the popup
    cost_mb: int  # resident cost (jetson), for the governor
    cost_measured: bool
    profiles: tuple[str, ...]
    dim: int | None = None  # embedder slots
    preprocess: Preprocess | None = None  # image_embed only
    prompt_template: str | None = None  # caption/planner only
    # Which languages a VOICE entry can speak (design §4.1, §11.1). Empty for every
    # other slot. `POST /speak` 503s when the film's language is not in here: a voice
    # reading Russian text through an English frontend is worse than silence.
    languages: tuple[str, ...] = ()
    # The voices this entry can read in (§11.1). Empty for every other slot.
    speakers: tuple[Speaker, ...] = ()
    # The entry speaks a SAVED voice from its example clip (§4.1, §11.1) instead of
    # preset speakers, so it declares none. Preset routing never picks it.
    clones: bool = False
    note: str | None = None
    adapter: AdapterSpec | None = None
    # The snapshot files the entry loads, fetched instead of the whole Hub repo. Empty
    # fetches the whole snapshot.
    files: tuple[str, ...] = ()
    # A face entry's declared use (§4.1, §8.1). None for every other slot.
    score_floor: float | None = None  # the lowest detection score kept
    match_threshold: float | None = None  # the lowest cosine similarity that joins a group
    input_edge: int | None = None  # the long edge, in px, the caller shrinks a photo to
    min_face_px: int | None = None  # the smallest face side kept, in px of that image


_ALL = ("mac", "jetson", "cloud")
_LOCAL = ("mac", "jetson")

CATALOG: tuple[ModelEntry, ...] = (
    # --- image_embed ---------------------------------------------------------
    ModelEntry(
        key="siglip2-so400m-384",
        slot="image_embed",
        display="SigLIP 2 so400m · 384px",
        source=HfSource("google/siglip2-so400m-patch14-384"),
        size_mb=4370,
        cost_mb=3400,
        cost_measured=True,
        profiles=_ALL,
        dim=1152,
        preprocess=Preprocess(384, "bilinear", "squash"),
        note="The default. Zero-shot tags, image↔text search and visual similarity.",
    ),
    ModelEntry(
        key="siglip2-so400m-512",
        slot="image_embed",
        display="SigLIP 2 so400m · 512px",
        # google publishes so400m at 512px as patch16, NOT patch14 — the patch14
        # repo does not exist and a download of it 404s.
        source=HfSource("google/siglip2-so400m-patch16-512"),
        size_mb=4373,
        cost_mb=3900,
        cost_measured=False,
        profiles=_ALL,
        dim=1152,
        preprocess=Preprocess(512, "bilinear", "squash"),
        note="A higher input resolution: finer detail, ~1.8x the embed time. Still "
        "1152-wide vectors, so switching does not rebuild the vector table — but "
        "every photo is still re-embedded.",
    ),
    # --- text_embed ----------------------------------------------------------
    ModelEntry(
        key="nomic-1.5",
        slot="text_embed",
        display="nomic-embed-text v1.5",
        source=HfSource("nomic-ai/nomic-embed-text-v1.5"),
        size_mb=2113,
        cost_mb=2200,
        cost_measured=True,
        profiles=_ALL,
        dim=768,
        note="The default. Needs `search_query:` / `search_document:` prefixes; "
        "loads with `.to()` (its remote code breaks under device_map), so it pays "
        "a host copy — 2.14 GB resident for a 274 MB model.",
    ),
    ModelEntry(
        key="embeddinggemma-300m",
        slot="text_embed",
        display="EmbeddingGemma 300M",
        source=HfSource("google/embeddinggemma-300m"),
        size_mb=1211,
        cost_mb=1600,
        cost_measured=False,
        profiles=_ALL,
        dim=768,
        note="Built for on-device retrieval: same width as nomic, a smaller "
        "resident set, so the conveyor evicts SigLIP less often.",
    ),
    ModelEntry(
        key="qwen3-embed-0.6b",
        slot="text_embed",
        display="Qwen3-Embedding 0.6B",
        source=HfSource("Qwen/Qwen3-Embedding-0.6B"),
        size_mb=1152,
        cost_mb=2400,
        cost_measured=False,
        profiles=_ALL,
        dim=1024,
        note="Strongest retrieval of the three and the most expensive resident.",
    ),
    # --- caption (the llama-server VISION mode) ------------------------------
    ModelEntry(
        key="gemma4-E2B",
        slot="caption",
        display="Gemma 4 E2B · vision",
        source=GgufSource(
            repo="unsloth/gemma-4-E2B-it-GGUF",
            file="gemma-4-E2B-it-Q4_K_M.gguf",
            mmproj_repo="ggml-org/gemma-4-E2B-it-GGUF",
            mmproj_file="mmproj-gemma-4-E2B-it-Q8_0.gguf",
        ),
        size_mb=3495,
        cost_mb=4300,
        cost_measured=True,
        profiles=_LOCAL,
        prompt_template="gemma4-E2B",
        note="The default. Q4_K_M weights + the Q8_0 projector; the same GGUF the "
        "planner slot uses, so keeping both here costs only the projector.",
    ),
    ModelEntry(
        key="qwen3-vl-4b",
        slot="caption",
        display="Qwen3-VL 4B Instruct",
        source=GgufSource(
            repo="unsloth/Qwen3-VL-4B-Instruct-GGUF",
            file="Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
            mmproj_repo="unsloth/Qwen3-VL-4B-Instruct-GGUF",
            mmproj_file="mmproj-F16.gguf",
        ),
        size_mb=3179,
        cost_mb=4200,
        cost_measured=False,
        profiles=_LOCAL,
        prompt_template="qwen3-vl",
        note="Noticeably better at dense scenes, text in the image, and spatial "
        "detail than gemma E2B — and slower per photo. Weights + the F16 projector; "
        "the weights are the same file the planner slot uses.",
    ),
    ModelEntry(
        key="qwen3-vl-8b",
        slot="caption",
        display="Qwen3-VL 8B Instruct",
        source=GgufSource(
            repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
            file="Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
            mmproj_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
            mmproj_file="mmproj-F16.gguf",
        ),
        size_mb=5900,
        cost_mb=7000,
        cost_measured=False,
        profiles=("mac",),
        prompt_template="qwen3-vl",
        note="Mac only — it does not fit the Jetson's budget alongside anything. "
        "Weights + the F16 projector; the weights are the same file the planner "
        "slot uses, so picking it in both slots downloads them once.",
    ),
    ModelEntry(
        key="qwen2.5vl-7b",
        slot="caption",
        display="Qwen2.5-VL 7B (vLLM)",
        source=HfSource("Qwen/Qwen2.5-VL-7B-Instruct"),
        size_mb=16000,
        cost_mb=20000,
        cost_measured=False,
        profiles=("cloud",),
        prompt_template="qwen2.5vl:7b",
        note="Served by vLLM; the cloud profile does not switch models from the UI.",
    ),
    # --- planner (the llama-server TEXT mode) --------------------------------
    ModelEntry(
        key="gemma4-E2B",
        slot="planner",
        display="Gemma 4 E2B · text",
        source=GgufSource(
            repo="unsloth/gemma-4-E2B-it-GGUF",
            file="gemma-4-E2B-it-Q4_K_M.gguf",
        ),
        size_mb=2963,
        cost_mb=3800,
        cost_measured=True,
        profiles=_LOCAL,
        prompt_template="gemma4-E2B",
        note="The default. Same GGUF as the caption slot, without the projector.",
    ),
    ModelEntry(
        key="qwen3-4b-2507",
        slot="planner",
        display="Qwen3 4B Instruct 2507",
        source=GgufSource(
            repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            file="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        ),
        size_mb=2382,
        cost_mb=3400,
        cost_measured=False,
        profiles=_LOCAL,
        prompt_template="qwen3",
        note="Better instruction-following and tool selection for the planner and "
        "the chat agent. Text only — pairing it with a vision caption model means "
        "llama-server restarts between the two roles.",
    ),
    ModelEntry(
        key="qwen3-vl-8b",
        slot="planner",
        display="Qwen3-VL 8B Instruct",
        source=GgufSource(
            repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
            file="Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        ),
        size_mb=4795,
        cost_mb=6500,
        cost_measured=False,
        profiles=("mac",),
        prompt_template="qwen3-vl",
        note="Mac only. Pair it with the same model in the caption slot and the "
        "server restarts only for the projector, as gemma does today.",
    ),
    ModelEntry(
        key="qwen2.5-3b",
        slot="planner",
        display="Qwen2.5 3B (vLLM)",
        source=HfSource("Qwen/Qwen2.5-3B-Instruct"),
        size_mb=7000,
        cost_mb=9000,
        cost_measured=False,
        profiles=("cloud",),
        prompt_template="qwen2.5:3b",
        note="Served by vLLM; the cloud profile does not switch models from the UI.",
    ),
    # --- voice (feed narration, design §11.1) --------------------------------
    ModelEntry(
        key="kokoro-82m",
        slot="voice",
        display="Kokoro 82M",
        source=HfSource("hexgrad/Kokoro-82M"),
        size_mb=330,
        cost_mb=1200,
        cost_measured=False,
        profiles=_ALL,
        languages=("en",),
        speakers=(
            Speaker("af_heart", "female", "warm",
                    "warm, close, unhurried — the default for an ordinary day"),
            Speaker("af_bella", "female", "bright",
                    "bright and lively, smiling; good for celebrations"),
            Speaker("af_sky", "female", "bright", "light and young"),
            Speaker("af_nicole", "female", "neutral", "soft and even, almost whispered"),
            Speaker("af_sarah", "female", "neutral", "plain and steady, unshowy"),
            Speaker("bf_emma", "female", "warm", "British, warm and measured"),
            Speaker("am_adam", "male", "neutral", "plain and steady"),
            Speaker("am_puck", "male", "bright", "playful and quick"),
            Speaker("am_michael", "male", "deep",
                    "low and grave; the darkest voice here"),
            Speaker("bm_george", "male", "deep", "British, low and dry"),
        ),
        note="The default. 82M params, Apache-2.0, 24 kHz mono, 54 voices. Reads a "
        "memory's title, description and captions over the feed. Needs the "
        "`espeak-ng` system package — its G2P shells out to it.",
    ),
    ModelEntry(
        key="silero-v4-ru",
        slot="voice",
        display="Silero v4 · Russian",
        source=UrlSource("https://models.silero.ai/models/tts/ru/v4_ru.pt", "v4_ru.pt"),
        size_mb=60,
        cost_mb=700,
        cost_measured=False,
        profiles=_ALL,
        languages=("ru",),
        speakers=(
            Speaker("xenia", "female", "warm", "warm and calm; the everyday choice"),
            Speaker("kseniya", "female", "bright", "brighter and livelier"),
            Speaker("baya", "female", "neutral", "even and unhurried"),
            Speaker("eugene", "male", "neutral", "plain male, mid register"),
            Speaker("aidar", "male", "deep",
                    "low and rough; the darkest Russian voice here"),
        ),
        note="Russian narration. Kokoro speaks nine languages but NOT Russian, so this "
        "is a second voice rather than a bigger one. One torch.package archive, pure "
        "torch — no espeak, no uroman — and tiny. NOT on Hugging Face: it is fetched "
        "straight from models.silero.ai. Licence is CC-BY-NC (Silero's MIT covers only "
        "their base cis-tts models): fine for a personal library, not for shipping.",
    ),
    ModelEntry(
        key="voxcpm2", slot="voice", display="VoxCPM2 · voice copy",
        source=HfSource("openbmb/VoxCPM2", revision="32279effe8c19989596f05d353d1447f51d9e915"),
        size_mb=4728, cost_mb=10500, cost_measured=False,
        profiles=("mac", "cloud"), languages=("en", "ru"), clones=True,
        note="Voice copy (§4.1, §11.1): speaks a SAVED voice from its example clip; no preset speakers. "
             "2.29B params, Apache-2.0 code and weights, 48 kHz mono. Pinned to one Hub commit; loaded with "
             "load_denoiser=False and no timestamps extra. MPS runs it in float32 (the package forces it), so "
             "the cost is an ESTIMATE: 2,290,004,544 x 4 B + the 377 MB AudioVAE + runtime. Measure it and pin it "
             "with IVMS777_MODEL_COST_MB. The voice child is built with LoRA layers for baked voices: about 18 MB "
             "in float32 (an estimate), inside cost_mb 10500. With no adapter loaded they add no adapter "
             "contribution; their extra matmuls are not measured.",
        adapter=AdapterSpec(
            lora={"enable_lm": True, "enable_dit": True, "enable_proj": False,
                  "r": 8, "alpha": 16, "dropout": 0.0,
                  "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
                  "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]},
            train_cost_mb=14000, train_cost_measured=False, train_profiles=("mac",),
            guidance_minutes=(5.0, 10.0),
        ),
    ),
    # --- music (feed soundtrack, design §11.1) -------------------------------
    ModelEntry(
        key="musicgen-small",
        slot="music",
        display="MusicGen small",
        source=HfSource("facebook/musicgen-small"),
        size_mb=2200,
        cost_mb=2600,
        cost_measured=False,
        profiles=_ALL,
        note="300M params, 32 kHz mono, generated autoregressively so any length is "
        "possible (capped at 30 s for cost). Weights are CC-BY-NC 4.0 — personal use "
        "only, not a commercial build; the code is MIT. Chosen over Stable Audio Open, "
        "which is a better model on paper but whose SDE sampler drives torchsde into "
        "non-terminating recursion on this torch/Python stack. Its cost is an ESTIMATE "
        "— measure it on the board and pin it with IVMS777_MODEL_COST_MB.",
    ),
    ModelEntry(
        key="musicgen-medium",
        slot="music",
        display="MusicGen medium",
        source=HfSource("facebook/musicgen-medium"),
        size_mb=7670,
        cost_mb=8200,
        cost_measured=False,
        profiles=("mac",),
        note="1.5B params, 32 kHz mono, the mac default since the score became 2-4 "
        "generated sections rather than one looped clip (§11.1). It is not slower: "
        "MEASURED on an M-series mac it generates at 4.75x realtime (14.9 s of audio "
        "in 71 s) against musicgen-small's 4.90x — autoregressive decode at this size "
        "is bound by memory bandwidth, not by parameter count, so the bigger model is "
        "very nearly free per second of music. What it costs is RESIDENT SIZE (fp32 "
        "weights are a 7.6 GB download) and a slower first load, which is why jetson "
        "and cloud stay on small — the 8 GB board has no room for it at all. Weights "
        "are CC-BY-NC 4.0, personal use only; the code is MIT. Its resident cost is an "
        "ESTIMATE — measure it and pin it with IVMS777_MODEL_COST_MB.",
    ),
)

# No slot and no user choice: it hears each take right after it is saved (§11.1).
TRANSCRIBERS: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="whisper-large-v3-turbo", slot="asr", display="Whisper large-v3 turbo · words of a take",
        source=HfSource("openai/whisper-large-v3-turbo",
                        revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
        size_mb=1547, cost_mb=4000, cost_measured=False,
        profiles=("mac", "cloud"), languages=("en", "ru"),
        note="Hears the words of each take of a saved voice right after it is saved (§4.1, §8.1, §11.1). "
             "808,878,080 params, MIT weights, 16 kHz input, a 30 s window (a take is 5-30 s). Loaded in "
             "float32 on MPS and CUDA, so the cost is an ESTIMATE: 808,878,080 x 4 B + runtime. Measure it "
             "and pin it with IVMS777_MODEL_COST_MB {\"asr\": ...}.",
    ),
)

# No slot and no user choice: it measures a test against the takes (§11.1) and speaks
# nothing. Not `SPEAKERS`: `ModelEntry.speakers` means preset narrators.
SPEAKER_EMBEDDERS: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="ecapa-voxceleb", slot="speaker_embed", display="ECAPA-TDNN VoxCeleb · speaker similarity",
        source=HfSource("speechbrain/spkrec-ecapa-voxceleb",
                        revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"),
        size_mb=85, cost_mb=1000, cost_measured=False,
        profiles=("mac",), dim=192,
        note="Compares a voice test with the voice's own takes (§4.1, §8.1, §11.1); a film never calls "
             "it. Apache-2.0 weights, 16 kHz input, a 192-value embedding. Its Russian accuracy is not "
             "measured. The cost is an ESTIMATE: 85 MiB of files + runtime. Measure it and pin it with "
             "IVMS777_MODEL_COST_MB {\"speaker_embed\": ...}.",
    ),
)

# No slot and no user choice: it finds and describes the faces of each photo (§4.1, §8).
FACE_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="buffalo-l", slot="faces", display="InsightFace buffalo_l · faces",
        source=HfSource("immich-app/buffalo_l",
                        revision="d09715916a0778919a770c343533641e250b8699"),
        files=("detection/model.onnx", "recognition/model.onnx"),
        size_mb=182, cost_mb=1000, cost_measured=False,
        profiles=_ALL, dim=512,
        score_floor=0.7, match_threshold=0.5, input_edge=640, min_face_px=32,
        note="Finds the faces of a photo and gives each one a 512-value vector (§4.1, §8.1): "
             "SCRFD-10G detection and ArcFace R50 recognition, ONNX converted to torch. The "
             "InsightFace weights are NON-COMMERCIAL, personal use only. The cost is an ESTIMATE: "
             "182 MiB of files + runtime. Measure it and pin it with "
             "IVMS777_MODEL_COST_MB {\"faces\": ...}.",
    ),
)

# Per-profile defaults. These reproduce `config.PROFILE_DEFAULTS` exactly: an
# untouched install must behave as it did before slots existed.
DEFAULTS: dict[str, dict[str, str]] = {
    "mac": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "gemma4-E2B",
        "planner": "gemma4-E2B",
        "voice": "kokoro-82m",
        # mac has the memory and pays no time for it — see the entry's measured note.
        "music": "musicgen-medium",
    },
    "jetson": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "gemma4-E2B",
        "planner": "gemma4-E2B",
        "voice": "kokoro-82m",
        "music": "musicgen-small",
    },
    "cloud": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "qwen2.5vl-7b",
        "planner": "qwen2.5-3b",
        "voice": "kokoro-82m",
        "music": "musicgen-small",
    },
}

_BY_SLOT_KEY = {(entry.slot, entry.key): entry for entry in CATALOG}


def get(slot: str, key: str) -> ModelEntry:
    """The entry for a slot's key. Slots are separate namespaces, so the same key
    can mean two things (`gemma4-E2B` is the text mode under `planner` and the
    vision mode under `caption`)."""
    return _BY_SLOT_KEY[(slot, key)]


def entries_for(slot: str, profile: str) -> list[ModelEntry]:
    """Everything offerable for a slot on a profile, default first."""
    default = DEFAULTS[profile][slot]
    offered = [e for e in CATALOG if e.slot == slot and profile in e.profiles]
    return sorted(offered, key=lambda e: (e.key != default, e.display))


def default_key(slot: str, profile: str) -> str:
    return DEFAULTS[profile][slot]


def cloning_entry_for(profile: str, language: str) -> ModelEntry | None:
    """The `voice` entry that speaks a SAVED voice in `language` on `profile`, or None
    (§11.1)."""
    for entry in entries_for("voice", profile):
        if entry.clones and language in entry.languages:
            return entry
    return None


def adapter_entry_for(profile: str) -> ModelEntry | None:
    """The cloning entry whose `AdapterSpec` trains on `profile`, or None (§4.1)."""
    for entry in entries_for("voice", profile):
        if entry.clones and entry.adapter is not None and profile in entry.adapter.train_profiles:
            return entry
    return None


def transcriber_for(profile: str) -> ModelEntry | None:
    """The `TRANSCRIBERS` entry offered on `profile`, or None (§4.1)."""
    for entry in TRANSCRIBERS:
        if profile in entry.profiles:
            return entry
    return None


def speaker_embedder_for(profile: str) -> ModelEntry | None:
    """The `SPEAKER_EMBEDDERS` entry offered on `profile`, or None (§4.1)."""
    for entry in SPEAKER_EMBEDDERS:
        if profile in entry.profiles:
            return entry
    return None


def face_model_for(profile: str) -> ModelEntry | None:
    """The `FACE_MODELS` entry offered on `profile`, or None (§4.1)."""
    for entry in FACE_MODELS:
        if profile in entry.profiles:
            return entry
    return None


def cloning_languages(profile: str) -> tuple[str, ...]:
    """The film languages a saved voice can narrate on `profile`, in the order the
    feed lists its languages."""
    # Imported here: the models image ships `models/catalog.py` without `feed/`
    # (Dockerfile.models), and only `app` asks this question.
    from feed.language import LANGUAGES

    return tuple(lang for lang in LANGUAGES if cloning_entry_for(profile, lang) is not None)


# `voice` is never offered to the user. Which voice narrates a film follows from the
# film's LANGUAGE and the memory's mood (§11.1) — nobody should have to know that
# Kokoro cannot speak Russian in order to get Russian narration.
AUTOMATIC_SLOTS: frozenset[str] = frozenset({"voice"})


def is_switchable(slot: str, profile: str) -> bool:
    """`cloud` serves one model per container (vLLM), so its slots are config-only
    (design §4.1); `voice` is automatic everywhere. Every other slot is switchable."""
    return profile != "cloud" and slot not in AUTOMATIC_SLOTS


def has(slot: str, key: str) -> bool:
    return (slot, key) in _BY_SLOT_KEY
