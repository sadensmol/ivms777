"""`RemoteEmbedder` — an `embedding.base.Embedder` implemented over HTTP via
`ModelsClient` (design §5.1, plan 15 task 2).

`config.Settings.build_embedder()`'s non-fake path returns this. It is a
transparent shim: `search/`, `ingest/`, `web/`, `chat/` keep calling the
`Embedder` protocol (`embed_images`, `embed_texts`, `.logit_scale`,
`.logit_bias`) exactly as before — they don't know or care that the actual
image embedder now lives in the `models` service, not in this process. This
module holds NO torch import; it only calls `ModelsClient`.

**The resize follows the MODEL, not a constant** (design §4.1). `GET /embed/spec`
returns the selected `image_embed` entry's preprocessing contract along with the
calibration, and this caches it until the slots' `generation` moves:

- `mode: "squash"` — stretch to `input_px`², ignoring aspect ratio. That is exactly
  what a fixed-resolution SigLIP processor does to any input it is given
  (`do_resize`, `size: {384, 384}`, `resample: BILINEAR`), so doing it here makes
  the processor's own resize an identity op and the vectors come out unchanged.
- `mode: "native"` — fit inside `input_px` with the aspect ratio kept, for NaFlex
  and other native-resolution encoders where squashing would throw away real signal.

Resizing here at all is the difference between a working ingest and a stalled one:
PNG is lossless, so encoding a 3024x4032 original cost 5.7 s of CPU and a 15 MB
base64 body PER PHOTO on the Jetson, against ~10 ms of GPU — the embed stage ran at
0.32 img/s with the GPU idle 96 % of the time (§8.1). At 384x384 the encode is a few
ms and the body is ~250 KB.
"""

from io import BytesIO

from PIL import Image

from inference.models_client import ModelsClient

_RESAMPLE = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


class RemoteEmbedder:
    def __init__(self, client: ModelsClient, model_name: str) -> None:
        self._client = client
        self.model_name = model_name
        self._spec: dict | None = None

    # --- the model's own contract, fetched once and cached -------------------
    def _ensure_spec(self) -> dict:
        # Fetched lazily and cached — the taxonomy pipeline (§9) reads
        # logit_scale/logit_bias per tag dimension, so this must not be one HTTP
        # round trip per access.
        if self._spec is None:
            self._spec = self._client.embed_spec()
        return self._spec

    def refresh(self) -> None:
        """Drop the cached spec — call after a slot switch (design §4.1)."""
        self._spec = None

    def _resized(self, image: Image.Image) -> Image.Image:
        pre = self._ensure_spec()["preprocess"]
        px = pre["input_px"]
        resample = _RESAMPLE.get(pre["resample"], Image.Resampling.BILINEAR)
        if pre["mode"] == "native":
            # Aspect-preserving: never upscale a smaller original either — a native
            # encoder consumes what it is given, and inventing pixels adds no signal.
            fitted = image.copy()
            fitted.thumbnail((px, px), resample)
            return fitted
        return image.resize((px, px), resample)

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        encoded = []
        for image in images:
            buffer = BytesIO()
            self._resized(image).save(buffer, "PNG")
            encoded.append(buffer.getvalue())
        return self._client.embed_image(encoded)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_text(texts)

    @property
    def logit_scale(self) -> float:
        return self._ensure_spec()["logit_scale"]

    @property
    def logit_bias(self) -> float:
        return self._ensure_spec()["logit_bias"]

    @property
    def generation(self) -> int:
        return self._ensure_spec()["generation"]
