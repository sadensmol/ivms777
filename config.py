from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["mac", "jetson", "cloud"]

# Real, currently-pullable model tags. Override per-deploy with IVMS777_CAPTION_MODEL
# / IVMS777_PLANNER_MODEL. Caption models must be vision-capable.
PROFILE_DEFAULTS: dict[Profile, dict[str, object]] = {
    "mac": {
        "caption_model": "qwen2.5vl:7b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cpu",
        "inference_base_url": "http://host.docker.internal:11434/v1",
        "ram_budget_mb": 24000,
        "caption_backend": "ollama",
    },
    "jetson": {
        "caption_model": "qwen2.5vl:3b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:11434/v1",
        "ram_budget_mb": 6000,
        # Ollama's CUDA build runs vision on CPU on JP7 (no Orin sm_87 vision
        # kernels), so jetson captions in-process instead (design §3.1/§4/§8.1).
        "caption_backend": "inprocess",
        "caption_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "cloud": {
        # vLLM serves whatever VLLM_MODEL is set to; point these at that model.
        "caption_model": "qwen2.5vl:7b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8000/v1",
        "ram_budget_mb": 60000,
        "caption_backend": "ollama",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IVMS777_", extra="ignore")

    profile: Profile = "mac"
    data_dir: Path = Path("/data")

    caption_model: str | None = None
    planner_model: str | None = None
    # Text-embedding model for caption semantics (§9); defaults to the planner
    # model, which is already loaded — no extra model to pull.
    text_embed_model: str | None = None
    embed_device: Literal["cpu", "cuda", "mps"] | None = None
    inference_base_url: str | None = None
    # Captioner adapter selection (design §4, §8.1): "ollama" (mac/cloud, today's
    # path) vs "inprocess" (jetson — transformers 4-bit VLM on the cu132 GPU,
    # because Ollama's CUDA build runs vision on CPU on JP7). caption_model_id is
    # the HF repo id used only by the inprocess backend.
    caption_backend: Literal["ollama", "inprocess"] | None = None
    caption_model_id: str | None = None

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

        Defaults to the real SigLIP; tests and the fast path set
        `use_fake_embedder`. Imports are local so torch loads only when the real
        model is actually built.
        """
        if self.use_fake_embedder:
            from embedding.fakes import FakeEmbedder

            return FakeEmbedder(), "fake"
        from embedding.siglip import get_siglip_embedder

        # Cached: the model loads once and is reused, so a search or a /photo click
        # never reloads ~400M of weights (config.build_embedder was doing exactly
        # that on every request).
        return get_siglip_embedder(self.embed_model_name, self.embed_device), self.embed_model_name

    @property
    def caption_embed_model(self) -> str:
        """Model used to embed caption text (§9) — the planner model by default,
        already resident, so no extra pull."""
        return self.text_embed_model or self.planner_model or "fake"

    def build_inference_client(self):
        """Return (client, caption_model).

        Real `OpenAICompatClient` against `inference_base_url` normally; tests set
        `use_fake_inference` to get an empty `FakeInferenceClient` and never touch
        a backend. The caption model is the (real) `caption_model` config value.
        """
        if self.use_fake_inference:
            from inference.fakes import FakeInferenceClient

            return FakeInferenceClient([]), self.caption_model or "fake"
        from inference.client import OpenAICompatClient

        return OpenAICompatClient(self.inference_base_url or ""), self.caption_model

    def build_captioner(self, client):
        """Return a `Captioner` (design §4, §8.1) for the profile's `caption_backend`.

        mac/cloud select `OllamaCaptioner` over `client` (today's caption path,
        byte-identical); jetson selects `VLMCaptioner`, CONSTRUCTED ONLY — no
        `.load()` here, so no torch import and no weight download. Loading is the
        model coordinator's job, under the `INGEST_CAPTION` lease. `use_fake_inference`
        always forces the Ollama adapter over a `FakeInferenceClient`, on every
        profile, so the offline test path stays reachable on jetson too. Imports
        are local so importing config never pulls in torch/transformers.
        """
        if self.use_fake_inference:
            from captioning.ollama_adapter import OllamaCaptioner
            from inference.fakes import FakeInferenceClient

            return OllamaCaptioner(FakeInferenceClient([]), "fake")
        if self.caption_backend == "inprocess":
            from captioning.vlm_adapter import VLMCaptioner

            return VLMCaptioner(self.caption_model_id, device=self.embed_device or "cuda")
        from captioning.ollama_adapter import OllamaCaptioner

        return OllamaCaptioner(client, self.caption_model or "fake")


@lru_cache
def get_settings() -> Settings:
    return Settings()
