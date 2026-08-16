"""`RemoteEmbedder` — an `embedding.base.Embedder` implemented over HTTP via
`ModelsClient` (design §5.1, plan 15 task 2).

`config.Settings.build_embedder()`'s non-fake path returns this. It is a
transparent shim: `search/`, `ingest/`, `web/`, `chat/` keep calling the
`Embedder` protocol (`embed_images`, `embed_texts`, `.logit_scale`,
`.logit_bias`) exactly as before — they don't know or care that the actual
SigLIP model now lives in the `models` service, not in this process. This
module holds NO torch import; it only calls `ModelsClient`.
"""

from io import BytesIO

from PIL import Image

from inference.models_client import ModelsClient


class RemoteEmbedder:
    def __init__(self, client: ModelsClient, model_name: str) -> None:
        self._client = client
        self.model_name = model_name
        self._logit_scale: float | None = None
        self._logit_bias: float | None = None

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        encoded = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, "PNG")
            encoded.append(buffer.getvalue())
        return self._client.embed_image(encoded)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_text(texts)

    def _ensure_calibration(self) -> None:
        # Fetched once, lazily, and cached — the taxonomy pipeline (§9) reads
        # logit_scale/logit_bias per tag dimension, so this must not be one
        # HTTP round trip per access.
        if self._logit_scale is None or self._logit_bias is None:
            calibration = self._client.calibration()
            self._logit_scale = calibration["logit_scale"]
            self._logit_bias = calibration["logit_bias"]

    @property
    def logit_scale(self) -> float:
        self._ensure_calibration()
        return self._logit_scale

    @property
    def logit_bias(self) -> float:
        self._ensure_calibration()
        return self._logit_bias
