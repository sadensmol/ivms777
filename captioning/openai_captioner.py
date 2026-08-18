"""Captioning over the OpenAI-compatible inference backend (design §4).

`llama-server` (mac + jetson) and vLLM (cloud) all speak the OpenAI
`/v1/chat/completions` API, so ONE adapter captions everywhere: it POSTs the
image as an `image_url` data-URI plus the caption prompt and constrains the
answer to `CAPTION_SCHEMA`. There is no per-profile branch and no in-process
vision model any more (plan 16 dropped the Jetson `transformers` VLM).

The model is server-resident (llama-server holds the one GGUF loaded for text
AND vision), so there is nothing to warm or evict here — captioning is a plain
HTTP call.
"""

import json

from captioning.base import CaptionResult
from inference.client import encode_image
from inference.prompts import CAPTION_SCHEMA, caption_messages


class OpenAICaptioner:
    """Captioning over the OpenAI-compatible inference backend (llama-server on
    mac/jetson, vLLM on cloud) — one image request per photo (design §4)."""

    def __init__(self, client, model: str, *, prompt_key: str | None = None):
        self._client = client
        self.name = model
        # The name STORED with every caption and sent as the OpenAI `model` field.
        self.caption_model = model
        # Which prompt template to use (`models/catalog.py`'s `prompt_template`,
        # design §4.1). Defaults to the model name, which is what it was before
        # slots existed.
        self.prompt_key = prompt_key or model

    def caption(self, image: bytes) -> CaptionResult:
        msgs = caption_messages(self.prompt_key, encode_image(image))
        raw = self._client.complete(self.caption_model, msgs, json_schema=CAPTION_SCHEMA)
        obj = json.loads(raw)
        return CaptionResult(obj["caption"], obj["title"], obj["description"])
