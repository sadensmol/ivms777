from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["mac", "jetson", "cloud"]

# Model ids per profile. On mac/jetson `llama-server` serves ONE gemma GGUF for
# BOTH text (planner/chat) and vision (caption) over the OpenAI `/v1` API, so
# both names point at the same model (plan 16, design §3.1/§4/§5.1). The name is
# passed through for storage/display; `llama-server` serves whatever `-m` loaded.
# Override per-deploy with IVMS777_CAPTION_MODEL / IVMS777_PLANNER_MODEL.
PROFILE_DEFAULTS: dict[Profile, dict[str, object]] = {
    "mac": {
        # One gemma4-E2B GGUF on a host-native llama-server (Metal), reached from
        # the containers at host.docker.internal:8080 (design §3.1, plan 16).
        "caption_model": "gemma4-E2B",
        "planner_model": "gemma4-E2B",
        "embed_device": "cpu",
        "inference_base_url": "http://host.docker.internal:8080/v1",
        "ram_budget_mb": 24000,
        "models_base_url": "http://models:9000",
    },
    "jetson": {
        # One gemma4-E2B GGUF on a containerised sm_87 CUDA llama-server, text +
        # vision on the GPU (design §3.1, plan 16). No Ollama, no in-process VLM.
        "caption_model": "gemma4-E2B",
        "planner_model": "gemma4-E2B",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8080/v1",
        "ram_budget_mb": 6000,
        "models_base_url": "http://models:9000",
    },
    "cloud": {
        # cloud is unchanged by plan 16 — still vLLM (open item).
        "caption_model": "qwen2.5vl:7b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8000/v1",
        "ram_budget_mb": 60000,
        "models_base_url": "http://models:9000",
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
    # "Similar photos": min image-vector cosine to count as a visual look-alike
    # (§9). SigLIP image cosines have a HIGH baseline — any two real photos sit
    # ~0.5–0.65 just for being photos, and genuinely-alike ones are 0.85–0.98 — so
    # the floor is 0.8, well above the noise. Tag/caption matches always qualify;
    # this only admits true look-alikes.
    similar_min_cosine: float = Field(default=0.8, ge=0.0, le=1.0)
    # "Similar photos": min caption-embedding cosine to count two captions as
    # semantically matching (§9). Tune per embedding model.
    similar_caption_min: float = Field(default=0.6, ge=0.0, le=1.0)

    embed_model_name: str = "siglip2-so400m-patch14-384"
    use_fake_embedder: bool = False
    use_fake_inference: bool = False

    # Usable RAM budget for the model coordinator (design §8.1). A workload whose
    # model-set exceeds this is refused, not loaded. Per-profile default below.
    ram_budget_mb: int | None = None

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
