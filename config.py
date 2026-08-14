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
    },
    "jetson": {
        "caption_model": "qwen2.5vl:3b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:11434/v1",
    },
    "cloud": {
        # vLLM serves whatever VLLM_MODEL is set to; point these at that model.
        "caption_model": "qwen2.5vl:7b",
        "planner_model": "qwen2.5:3b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8000/v1",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IVMS777_", extra="ignore")

    profile: Profile = "mac"
    data_dir: Path = Path("/data")

    caption_model: str | None = None
    planner_model: str | None = None
    embed_device: Literal["cpu", "cuda", "mps"] | None = None
    inference_base_url: str | None = None

    owner_id: int = 1
    thumb_grid_px: int = 320
    thumb_detail_px: int = 1600
    page_size: int = Field(default=100, ge=1, le=500)
    # Minimum tag score for a model tag to count in the sidebar and filters.
    tag_score_min: float = Field(default=0.2, ge=0.0, le=1.0)

    embed_model_name: str = "siglip2-so400m-patch14-384"
    use_fake_embedder: bool = False
    use_fake_inference: bool = False

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
        from embedding.siglip import SiglipEmbedder

        return SiglipEmbedder(self.embed_model_name, self.embed_device), self.embed_model_name

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
