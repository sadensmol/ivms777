import gc
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from embedding.base import EMBED_DIM

# Hugging Face id for SigLIP 2 so400m/14 at 384px. The short name in config maps
# here so config stays terse and the actual repo id lives in one place.
_HF_ID = "google/siglip2-so400m-patch14-384"


class SiglipEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.model = AutoModel.from_pretrained(_HF_ID).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(_HF_ID)
        # SigLIP's learned zero-shot calibration (see embedding.vectors); read from
        # the model so it is always exact for this checkpoint.
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = float(self.model.logit_bias.item())

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self._normalized(self.model.get_image_features(**inputs))

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        return self._normalized(self.model.get_text_features(**inputs))

    @staticmethod
    def _normalized(features) -> list[list[float]]:
        # transformers 5.x returns a BaseModelOutputWithPooling; the [n, 1152]
        # embedding is its pooler_output. Older/other builds return a bare tensor.
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        array = features.float().cpu().numpy()
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = array / norms
        assert unit.shape[1] == EMBED_DIM, f"expected {EMBED_DIM}, got {unit.shape[1]}"
        return unit.tolist()


@lru_cache(maxsize=2)
def get_siglip_embedder(model_name: str, device: str) -> SiglipEmbedder:
    """Load the SigLIP model once and reuse it.

    Building a `SiglipEmbedder` loads ~400M of weights from disk into torch — far
    too slow to redo per request. Every route (search, /photo ctx paging, chat)
    goes through this cache, so the model is resident after the first use and a
    photo click no longer reloads it.
    """
    return SiglipEmbedder(model_name, device)


def release_siglip_embedder() -> None:
    """Drop the cached SigLIP model and return its GPU allocation to the driver.

    On a unified-memory Jetson (8 GB shared CPU+GPU) SigLIP and the Ollama vision
    captioner cannot both be resident: a loaded SigLIP pins torch's CUDA caching
    allocator, so CUDA reports almost no free memory and Ollama silently offloads
    the captioner to the CPU (≈20× slower). The worker calls this after the
    embed/taxonomy stages and before the caption stage, so the captioner gets the
    GPU (design §8). `empty_cache()` is what actually hands the freed blocks back
    to the driver — clearing the cache only drops Python references.
    """
    get_siglip_embedder.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
