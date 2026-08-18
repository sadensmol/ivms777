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
        # One gemma4-E2B GGUF on a host-native llama-server (Metal), started by
        # `make llama-mac` on :8080 (design §3.1, plan 16). The models service does
        # NOT supervise it — it is an EXTERNAL server the service reuses (RemoteLlm),
        # so `llm_managed` is False; spawning a second llama-server would clash on
        # :8080. Docker Desktop has no GPU passthrough, hence host-native.
        "caption_model": "gemma4-E2B",
        "planner_model": "gemma4-E2B",
        "embed_device": "cpu",
        "inference_base_url": "http://host.docker.internal:8080/v1",
        "ram_budget_mb": 24000,
        "models_base_url": "http://models:9000",
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
    similar_min_cosine: float = Field(default=signals.IMAGE_GATE, ge=0.0, le=1.0)
    # Caption-embedding cosine. Two RANDOM captions score a median 0.621, so the old
    # 0.60 sat *below* the noise and 64% of all pairs cleared it — that is why a
    # teddy bear matched a girl by a Christmas tree. 0.75 is the top 5%.
    similar_caption_min: float = Field(default=signals.CAPTION_GATE, ge=0.0, le=1.0)
    # Minimum FINAL score (0–1, a noisy-OR of the evidence) to be shown at all (§9).
    # Without it the strip always returns its full k, so a photo with nothing
    # genuinely like it got 12 fillers instead of an honest "nothing similar enough".
    # Sized so a lone `moment` at same-hour/same-block (0.30) survives while
    # same-afternoon/500 m (0.23) needs a second signal to agree.
    similar_score_min: float = Field(default=signals.SCORE_MIN, ge=0.0, le=1.0)
    # Kill switch for the caption-meaning signal in "similar photos" (§9.3). False
    # drops it from BOTH halves — the candidate union and the scoring contribution —
    # leaving tags + image look-alike. It exists to measure what the caption signal
    # is actually worth (`scripts/caption_ablation.py`): if it earns nothing, the
    # nomic text embedder can be dropped from the resident set entirely (§8.1).
    similar_use_captions: bool = True

    embed_model_name: str = "siglip2-so400m-patch14-384"
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
    # Rough resident cost of each managed model, for the governor's budget math.
    # gemma is the containerised `llama-server` child: the Q4_K_M weights (~3.0 GB)
    # + the Q8_0 mmproj (~0.5 GB) + KV/compute buffers + its own CUDA context.
    # This MUST be accurate in BOTH directions:
    #   - an UNDER-estimate makes the governor keep SigLIP resident and load gemma
    #     on top → OOM kills the child → captions fail with "connection refused";
    #   - an OVER-estimate does NOT just "evict more, which is safe". Once
    #     `cost + headroom` exceeds `ram_budget_mb` the model can never load AT
    #     ALL, on any amount of free RAM — the governor stops evicting and starts
    #     refusing. `gemma-vision` sat at 5000 against a 5000 budget and a 512
    #     headroom, so `5000 + 512 > 5000` raised InsufficientMemory on every
    #     single caption; the whole stage failed on an idle board with 5.6 GB
    #     free. `test_every_model_fits_its_profile_budget` now guards that.
    # Both gemma figures are MEASURED on the board, like siglip/nomic — as the drop
    # in `psutil.virtual_memory().available`, which is exactly what the governor's
    # real-free guard reads, across loading each on an otherwise-idle Jetson:
    #   gemma        3606 MB (peak RAM delta 3374) → declared 3800
    #   gemma-vision 3936 MB (peak RAM delta 4047) → declared 4300
    # `llama-server`'s RSS reads ~4967 MB for gemma-vision, but ~1.3 GB of that is
    # the mmap'd GGUF already counted in page cache — RSS is the wrong unit here.
    # The ~500 MB gap between the two is the Q8_0 projector, which is what it
    # should be: the measurements corroborate each other.
    # `gemma` is the TEXT-only child (chat/planner); `gemma-vision` adds the Q8_0
    # projector for captioning (design §3.1). Mutually exclusive — never both, so
    # vision MUST cost more than text-only or the swap logic is describing a
    # model that does not exist.
    # siglip/nomic are MEASURED on the board, not guessed — re-measured after the
    # `device_map` fix (§8.1). Each figure is the CHILD's whole footprint (its torch
    # import and CUDA context included), which is the right unit because evicting
    # kills the child:
    #   siglip 3.26 GB — loads with `device_map`, so no host copy. Was ~5.0 GB against
    #     a declared 3400 before the fix: exactly the under-estimate this warns about.
    #   nomic  2.14 GB — still loads with `.to()` (device_map breaks its custom remote
    #     code), so it PAYS the host copy. Measured at 1.28 GB under device_map, which
    #     is what it would cost if that ever becomes usable. NOT the "~0.3 GB" older
    #     docs claim — nomic is the second-most expensive resident on this board.
    # Since plan 20 each runs in a `TorchWorker` child, so `free` is a process kill
    # and the eviction genuinely returns that memory to gemma (§8.1).
    model_cost_mb: dict[str, int] = Field(
        default_factory=lambda: {
            "siglip": 3400,
            "nomic": 2200,
            "gemma": 3800,
            "gemma-vision": 4300,
        }
    )
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
    mmproj_name: str = "mmproj-gemma-4-E2B-it-Q8_0.gguf"
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

    def build_embedder(self):
        """Return (embedder, model_name).

        Defaults to `RemoteEmbedder`, an HTTP shim over the `models` service
        (design §5.1) — the real SigLIP now lives there, never in this
        process. Tests and the fast path set `use_fake_embedder` to get the
        in-process `FakeEmbedder` instead. Imports are local so importing
        `config` never pulls in `httpx`/torch until a caller actually builds
        an embedder; this module itself never imports torch.
        """
        if self.use_fake_embedder:
            from embedding.fakes import FakeEmbedder

            return FakeEmbedder(), "fake"
        from inference.remote_embedder import RemoteEmbedder

        return RemoteEmbedder(self.build_models_client(), self.embed_model_name), self.embed_model_name

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
