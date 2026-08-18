from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from search import signals

Profile = Literal["mac", "jetson", "cloud"]

# Model ids per profile. On mac/jetson `llama-server` serves ONE gemma GGUF for
# BOTH text (planner/chat) and vision (caption) over the OpenAI `/v1` API, so
# both names point at the same model (plan 16, design §3.1/§4/§5.1). The name is
# passed through for storage/display; `llama-server` serves whatever `-m` loaded.
# Override per-deploy with IVMS777_CAPTION_MODEL / IVMS777_PLANNER_MODEL.
PROFILE_DEFAULTS: dict[Profile, dict[str, object]] = {
    "mac": {
        # EVERYTHING runs host-native on the Apple GPU — `make up` starts the
        # models service, the worker and the app as plain host processes (design
        # §3.1). Nothing model-related is containerised on mac, because Docker
        # Desktop has no Metal passthrough and a container can only fall back to
        # the CPU, which §3.1 forbids on every profile.
        #
        # One gemma4-E2B GGUF on a host-native llama-server (Metal), started by
        # `make llama-mac` on :8080 (plan 16). The models service does NOT
        # supervise it — it is an EXTERNAL server the service reuses (RemoteLlm),
        # so `llm_managed` is False; spawning a second llama-server would clash on
        # :8080. SigLIP and the caption-text embedder reach the same GPU through
        # torch's `mps` backend.
        "caption_model": "gemma4-E2B",
        "planner_model": "gemma4-E2B",
        "embed_device": "mps",
        "inference_base_url": "http://localhost:8080/v1",
        "ram_budget_mb": 24000,
        "models_base_url": "http://localhost:9000",
        "gpu_concurrency": 3,
        "llm_managed": False,
        "llm_idle_ttl_s": None,
        "llm_ngl": 99,
        "llm_ctx": 4096,
    },
    "jetson": {
        # One gemma4-E2B GGUF on a containerised sm_87 CUDA llama-server, text +
        # vision on the GPU (design §3.1, plan 16). No Ollama, no in-process VLM.
        "caption_model": "gemma4-E2B",
        "planner_model": "gemma4-E2B",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8080/v1",
        # 7.4 GB usable, and app+worker+OS hold ~1.9 GB of it (measured: app 647 MB,
        # worker 625 MB, host ~600 MB). A 6000 budget over-commits the board before
        # a single model loads — the governor thought it had headroom while `free`
        # showed 461 MB and 1.2 GB had already gone to swap.
        "ram_budget_mb": 5000,
        "models_base_url": "http://models:9000",
        "gpu_concurrency": 1,
        "llm_managed": True,
        "llm_idle_ttl_s": 120,
        # GPU-ONLY, ALWAYS: every layer on the GPU, never a CPU offload (design
        # §3.1/§8.1). gemma is made to FIT instead — the conveyor evicts SigLIP +
        # nomic, the KV context is 2048, and the vision projector is the Q8_0 build
        # (~0.5 GB vs ~0.94 GB F16). If it still does not fit, llama-server aborts
        # and the error is surfaced; we shrink the footprint, never spill to CPU.
        "llm_ngl": 99,
        # Smaller KV cache to shrink gemma's resident footprint on the 8 GB box
        # (~0.5 GB less than 4096) — chat turns are short (design §3.1/§8.1).
        "llm_ctx": 2048,
    },
    "cloud": {
        # cloud is unchanged by plan 16 — still vLLM (open item).
        "caption_model": "qwen2.5vl:7b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8000/v1",
        "ram_budget_mb": 60000,
        "models_base_url": "http://models:9000",
        "gpu_concurrency": 4,
        "llm_managed": False,
        "llm_idle_ttl_s": None,
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IVMS777_", extra="ignore")

    profile: Profile = "mac"
    data_dir: Path = Path("/data")

    # Slot overrides. Since plan 21 these are **catalog keys** (`models/catalog.py`),
    # not free-form model names: they are the fallback when nothing is stored in
    # `app_settings`, and `models/slots.py` ignores a value that is not an entry for
    # that slot on this profile (so a leftover `qwen2.5vl:7b` on cloud, or an old
    # Ollama tag, degrades to the profile default instead of failing to boot).
    caption_model: str | None = None
    planner_model: str | None = None
    # Dedicated text embedder for caption semantics (§9). A chat model cannot
    # embed (no embedding head), and SigLIP's text tower has no text↔text
    # separation (design §4), so this is a purpose-built text embedder —
    # `nomic-embed-text-v1.5` by default (benchmarked on real captions, §4). Since
    # plan 16 drops Ollama, it is loaded IN-PROCESS in the `models` service (mac +
    # jetson) via `transformers`, not served by a text backend (design §4/§5.1).
    text_embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embed_device: Literal["cpu", "cuda", "mps"] | None = None
    inference_base_url: str | None = None
    # Base URL of the `models` service (design §5.1, plan 15) — the one process
    # that imports torch/transformers and the only client of every inference
    # backend. `app`/`worker` reach every model/LLM through `build_models_client()`.
    models_base_url: str | None = None

    owner_id: int = 1
    thumb_grid_px: int = 320
    thumb_detail_px: int = 1600
    page_size: int = Field(default=100, ge=1, le=500)
    # Minimum tag score for a model tag to count in the sidebar and filters.
    tag_score_min: float = Field(default=0.2, ge=0.0, le=1.0)
    # "Similar photos" — the three GATES a caller may retune (§9). Weights and the
    # remaining gates live in `search/signals.py`, which also carries the measured
    # distribution behind each number. The one rule when changing these: NEVER set a
    # gate below the library's random-pair median, or the signal scores pure chance.
    #
    # Image cosine. SigLIP image cosines have a HIGH baseline — two RANDOM photos of
    # the reference library score a median 0.558 — so 0.80 (the top 4% of all pairs)
    # is the bar for a genuine look-alike.
    similar_min_cosine: float = Field(default=signals.STRICT.image, ge=0.0, le=1.0)
    # Caption-embedding cosine. Two RANDOM captions score a median 0.621, so the old
    # 0.60 sat *below* the noise and 64% of all pairs cleared it — that is why a
    # teddy bear matched a girl by a Christmas tree. 0.75 is the top 5%.
    similar_caption_min: float = Field(default=signals.STRICT.caption, ge=0.0, le=1.0)
    # Minimum FINAL score (0–1, a noisy-OR of the evidence) to be shown at all (§9).
    # Without it the strip always returns its full k, so a photo with nothing
    # genuinely like it got 12 fillers instead of an honest "nothing similar enough".
    # Measured against the old model over 22 seeds: below this the strip's tail
    # filled with moment-only pairs (a tire close-up matched a plastic container,
    # same minute, unrelated object). Those belong behind "Show more" (§9), not in
    # the default strip.
    similar_score_min: float = Field(default=signals.STRICT.score_min, ge=0.0, le=1.0)
    # Kill switch for the caption-meaning signal in "similar photos" (§9.3). False
    # drops it from BOTH halves — the candidate union and the scoring contribution —
    # leaving tags + image look-alike. It exists to measure what the caption signal
    # is actually worth (`scripts/caption_ablation.py`): if it earns nothing, the
    # nomic text embedder can be dropped from the resident set entirely (§8.1).
    similar_use_captions: bool = True

    # Env override for the `image_embed` slot — a CATALOG KEY (`siglip2-so400m-384`),
    # never a bare HF repo id, exactly like `caption_model`/`planner_model` (§4.1).
    # `None` means "use the profile default". It used to default to the HF id
    # `siglip2-so400m-patch14-384`, which is not a catalog key: the resolver ignored
    # it, so the override did nothing while the string still leaked to the UI.
    embed_model_name: str | None = None
    use_fake_embedder: bool = False
    use_fake_inference: bool = False

    # Whether `/api/upload/finish` drains the ingest queue INLINE, in the app
    # process (§5, §8). True is the single-process convenience: a bare `uvicorn`
    # run and every test get a usable, searchable grid straight after upload
    # without a `worker` to poll.
    #
    # It must be False wherever a `worker` container runs — which is every
    # compose profile. The drain is a FULL pass over the whole queue, so with a
    # worker present it (a) holds the finish request open for the entire
    # library — minutes for a few hundred photos — and (b) puts a second drainer
    # on the same jobs, doubling CPU and GPU contention. On the Jetson that
    # showed up as `app` pinned at 100 % CPU for the whole ingest, racing the
    # worker for a GPU that only has `gpu_concurrency: 1`. `compose.yaml` sets
    # it False for `app`; §5's "app serves reads, worker owns writes" is the
    # deployed behaviour.
    inline_drain: bool = True

    # Usable RAM budget for the model conveyor (design §8.1). The MemoryGovernor
    # evicts resident models to keep the resident set within this ceiling AND
    # within measured free RAM. Per-profile default below.
    ram_budget_mb: int | None = None

    # --- Single model conveyor (design §5.1/§8.1, plan 18) ---
    # Max GPU-heavy ops the scheduler runs at once: 1 on jetson (serialize the
    # GPU), N on mac/cloud (parallel). Per-profile default below.
    gpu_concurrency: int | None = None
    # Whether the `models` service supervises `llama-server` as a child process
    # (mac/jetson). False on cloud (remote vLLM, not supervised).
    llm_managed: bool | None = None
    # Unload gemma after this many idle seconds to free ~2 GB for SigLIP-heavy
    # batches (jetson). None disables idle-unload (mac/cloud). Per-profile default.
    llm_idle_ttl_s: int | None = None
    # Per-UNIT override of the resident cost the governor budgets against. Since
    # plan 21 the cost comes from the SELECTED catalog entry (`models/catalog.py`,
    # design §4.1); this map is the board-side escape hatch — set
    # `IVMS777_MODEL_COST_MB='{"llm_vision": 4500}'` to correct a figure without
    # editing the catalog. Keys are the four residency units: `image_embed`,
    # `text_embed`, `llm`, `llm_vision`. Empty by default: the catalog is the
    # source of truth, and its defaults carry the measurements below.
    #
    # A cost MUST be accurate in BOTH directions:
    #   - an UNDER-estimate makes the governor keep SigLIP resident and load gemma
    #     on top → OOM kills the child → captions fail with "connection refused";
    #   - an OVER-estimate does NOT just "evict more, which is safe". Once
    #     `cost + headroom` exceeds `ram_budget_mb` the model can never load AT
    #     ALL, on any amount of free RAM — the governor stops evicting and starts
    #     refusing. `gemma-vision` sat at 5000 against a 5000 budget and a 512
    #     headroom, so `5000 + 512 > 5000` raised InsufficientMemory on every
    #     single caption; the whole stage failed on an idle board with 5.6 GB
    #     free. `test_every_model_fits_its_profile_budget` now guards that.
    # The catalog's four default costs are MEASURED on the board — as the drop in
    # `psutil.virtual_memory().available`, which is exactly what the governor's
    # real-free guard reads, across loading each on an otherwise-idle Jetson. Each
    # figure is the CHILD's whole footprint (torch import and CUDA context
    # included), which is the right unit because evicting kills the child:
    #   llm (gemma text)   3606 MB (peak RAM delta 3374) → declared 3800
    #   llm_vision (+proj) 3936 MB (peak RAM delta 4047) → declared 4300
    #   image_embed        3.26 GB — loads with `device_map`, so no host copy
    #   text_embed         2.14 GB — nomic keeps `.to()` (device_map breaks its
    #     remote code) and PAYS the host copy; 1.28 GB if that ever becomes usable
    # `llama-server`'s RSS reads ~4967 MB for the vision mode, but ~1.3 GB of that
    # is the mmap'd GGUF already counted in page cache — RSS is the wrong unit. The
    # ~500 MB gap between the two llm modes is the Q8_0 projector, exactly as it
    # should be: the measurements corroborate each other. Vision MUST cost more than
    # text-only or the swap logic describes a model that does not exist.
    model_cost_mb: dict[str, int] = Field(default_factory=dict)
    # Path to the `llama-server` binary the models service spawns when it SUPERVISES
    # gemma (jetson: llm_managed=True). None → resolve `llama-server` from PATH. On
    # mac gemma runs as an EXTERNAL host server (make llama-mac), so this is unused.
    llm_bin: str | None = None
    llm_port: int = 8080
    # KV-cache context length for the supervised `llama-server` (`-c`). Smaller =
    # less resident RAM; jetson uses 2048 (chat turns are short) to give the 8 GB
    # box headroom, mac/cloud 4096. Per-profile default below (env: IVMS777_LLM_CTX).
    llm_ctx: int | None = None
    # The mmproj (vision projector) GGUF `llama-server` loads for captioning. Only
    # used where the service SUPERVISES gemma (jetson); mac reuses an external server.
    # Default is the **Q8_0** build (531 MB, `ggml-org/gemma-4-E2B-it-GGUF`) rather
    # than F16 (985 MB): the projector is GPU-resident like everything else (no CPU
    # offload, §3.1), and its CLIP tensor buffer is what OOM-aborted llama-server at
    # F16. Q8 costs negligible caption quality and is the single biggest GPU-RAM
    # saving on the 8 GB jetson. Override with IVMS777_MMPROJ_NAME.
    # None → use the projector the selected `caption` catalog entry names (§4.1).
    mmproj_name: str | None = None
    # GPU layers for the supervised `llama-server` (`-ngl`). None → omit the flag so
    # llama.cpp AUTO-FITS as many layers as free GPU RAM allows and offloads the rest
    # to CPU — the jetson safety net: gemma (~5 GB) can exceed the ~4.3 GB free, and
    # forcing `-ngl 99` makes llama.cpp ABORT at load ("failed to fit params …, abort")
    # instead of degrading, killing the child (design §3.1/§8.1). An int pins that many
    # GPU layers (mac: 99, Metal has ample RAM). Per-profile default below.
    llm_ngl: int | None = None

    @property
    def llm_health_url(self) -> str:
        return f"http://localhost:{self.llm_port}/health"

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "Settings":
        for key, value in PROFILE_DEFAULTS[self.profile].items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ivms777.db"

    @property
    def thumb_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    def embed_model_key(self, conn=None) -> str:
        """The name to record against a vector this process produces — the catalog
        key the `image_embed` slot holds right now (§4.1), or `"fake"` when the
        fake embedder is wired in (its vectors come from no model at all).

        Pass `conn` for the user's STORED choice; without it this is the env
        override → profile default, all a connection-less caller can know.
        """
        if self.use_fake_embedder:
            return "fake"
        from models import slots

        return slots.resolve_key(conn, self, "image_embed")

    def build_embedder(self, conn=None):
        """Return (embedder, model_name).

        Defaults to `RemoteEmbedder`, an HTTP shim over the `models` service
        (design §5.1) — the real SigLIP now lives there, never in this
        process. Tests and the fast path set `use_fake_embedder` to get the
        in-process `FakeEmbedder` instead. Imports are local so importing
        `config` never pulls in `httpx`/torch until a caller actually builds
        an embedder; this module itself never imports torch.

        `model_name` is the catalog key the `image_embed` slot holds RIGHT NOW
        (§4.1) — it is what the embed stage stamps on `photos.embedding_model`.
        Pass `conn` to see the user's STORED choice; without it the answer is
        the env override → profile default, which is all a connection-less
        caller can know. Reading `embed_model_name` directly is NOT the answer:
        it is only the env override that feeds the resolver, so the photo page
        claimed `siglip2-so400m-patch14-384` for every photo no matter which
        model was selected.
        """
        key = self.embed_model_key(conn)
        if self.use_fake_embedder:
            from embedding.fakes import FakeEmbedder

            return FakeEmbedder(), key
        from inference.remote_embedder import RemoteEmbedder

        return RemoteEmbedder(self.build_models_client(), key), key

    @property
    def caption_embed_model(self) -> str:
        """Dedicated text embedder for caption meaning (§9) — `nomic-embed-text-v1.5`
        by default, loaded in-process in the `models` service (design §4/§5.1). NOT
        the planner (a chat model can't embed) and NOT SigLIP (no text↔text
        separation). See `text_embed_model`."""
        return self.text_embed_model

    def build_inference_client(self):
        """Return (client, caption_model).

        Defaults to `RemoteInferenceClient`, an HTTP shim over the `models`
        service (design §5.1) — the real inference client (llama-server on
        mac/jetson, vLLM on cloud) now lives there, never in this process. Tests
        set `use_fake_inference` to get the in-process
        `FakeInferenceClient` instead. Imports are local so importing `config`
        never pulls in `httpx`/torch until a caller actually builds a client;
        this module itself never imports `OpenAICompatClient`.
        """
        if self.use_fake_inference:
            from inference.fakes import FakeInferenceClient

            return FakeInferenceClient([]), self.caption_model or "fake"
        from inference.remote_inference_client import RemoteInferenceClient

        return RemoteInferenceClient(self.build_models_client()), self.caption_model

    def build_models_client(self):
        """Return a `ModelsClient` for the `models` service (design §5.1).

        Import is local so importing `config` never pulls in `httpx`'s HTTP
        machinery until a caller actually needs the client.
        """
        from inference.models_client import ModelsClient

        return ModelsClient(self.models_base_url or "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
