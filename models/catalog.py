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

SLOTS: tuple[str, ...] = ("image_embed", "text_embed", "caption", "planner")

# Which ingest stages a slot switch invalidates, inclusive (design §4.1). `None`
# means the slot stores nothing, so switching it re-runs nothing.
INVALIDATES: dict[str, tuple[str, str] | None] = {
    "image_embed": ("embed", "taxonomy"),
    "text_embed": ("caption_embed", "caption_embed"),
    "caption": ("caption", "caption_embed"),
    "planner": None,
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
    """A `transformers` checkpoint, fetched into the HF cache."""

    repo: str


@dataclass(frozen=True)
class GgufSource:
    """A llama.cpp GGUF (+ optional vision projector), fetched into `data_dir/models`."""

    repo: str
    file: str
    mmproj_repo: str | None = None
    mmproj_file: str | None = None


@dataclass(frozen=True)
class ModelEntry:
    key: str
    slot: str
    display: str
    source: HfSource | GgufSource
    size_mb: int  # download size, for the popup
    cost_mb: int  # resident cost (jetson), for the governor
    cost_measured: bool
    profiles: tuple[str, ...]
    dim: int | None = None  # embedder slots
    preprocess: Preprocess | None = None  # image_embed only
    prompt_template: str | None = None  # caption/planner only
    note: str | None = None


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
)

# Per-profile defaults. These reproduce `config.PROFILE_DEFAULTS` exactly: an
# untouched install must behave as it did before slots existed.
DEFAULTS: dict[str, dict[str, str]] = {
    "mac": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "gemma4-E2B",
        "planner": "gemma4-E2B",
    },
    "jetson": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "gemma4-E2B",
        "planner": "gemma4-E2B",
    },
    "cloud": {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "qwen2.5vl-7b",
        "planner": "qwen2.5-3b",
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


def is_switchable(slot: str, profile: str) -> bool:
    """`cloud` serves one model per container (vLLM), so its slots are config-only
    (design §4.1). Everywhere else every slot is switchable from the UI."""
    return profile != "cloud"


def has(slot: str, key: str) -> bool:
    return (slot, key) in _BY_SLOT_KEY
