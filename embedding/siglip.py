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
        # fp16 on cuda: this ~1B-param model is ~4 GB in float32 but ~2 GB in
        # float16 — on the 8 GB unified Jetson that difference is the whole caption
        # swap budget (§8/§8.1). CPU/MPS stay float32 (half is slow/unsupported on
        # CPU). Inference-only, so fp16 precision is fine for cosine similarity.
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        # `device_map=device`, NOT `.to(device)`. `.to()` materialises every weight in
        # HOST RAM first and the copy is never released — measured on the Jetson at
        # 5128 MB RSS vs 2940 MB here, a 2.2 GB difference on a 7.4 GB board (§8.1).
        # `gc`/`empty_cache`/`malloc_trim` do NOT reclaim it; the tensors stay
        # referenced. `device_map` streams shard-by-shard straight to the device, so
        # the host copy never exists. Needs `accelerate` (the `models` extra).
        self.model = AutoModel.from_pretrained(
            _HF_ID, dtype=self.dtype, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(_HF_ID)
        # SigLIP's learned zero-shot calibration (see embedding.vectors); read from
        # the model so it is always exact for this checkpoint.
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = float(self.model.logit_bias.item())

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # pixel_values is float; match the model dtype (fp16 on cuda) or the conv
        # rejects it. Text ids stay integer, so embed_texts needs no cast.
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return self._normalized(self.model.get_image_features(**inputs))

    def embed_image_bytes(self, images: list[bytes]) -> list[list[float]]:
        """Decode, then embed. The bytes (not PIL objects) are what cross the
        `TorchWorker` pipe, so the decode happens here — in the child that owns
        torch — and the models parent stays image-library-free (plan 20)."""
        from io import BytesIO

        return self.embed_images([Image.open(BytesIO(b)).convert("RGB") for b in images])

    def calibration(self) -> dict:
        """SigLIP's zero-shot calibration, as the worker protocol returns it."""
        return {"logit_scale": self.logit_scale, "logit_bias": self.logit_bias}

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
