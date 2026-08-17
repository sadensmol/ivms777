"""Config -> `ModelBackend` factory (design §5.1).

`build_backend` is the profile-config entrypoint to a running `models`
service: the real service main calls
`create_models_app(build_backend(settings))`.

Ruling R1: `FakeBackend` iff `settings.use_fake_embedder` — that flag alone
picks the fake path, kept deliberately simple.
"""

from modelsvc.backends.base import ModelBackend
from modelsvc.backends.caption_backend import CaptionBackend
from modelsvc.backends.composite import CompositeBackend
from modelsvc.backends.fake import FakeBackend
from modelsvc.backends.siglip_backend import SiglipBackend
from modelsvc.backends.text_backend import TextBackend


def build_backend(settings) -> ModelBackend:
    if settings.use_fake_embedder:
        return FakeBackend()
    # Board-only: constructs but does not load anything (SiglipBackend and the
    # in-process text embedder are both lazy, and the captioner is a remote HTTP
    # call), so this branch only pulls torch/transformers when a request actually
    # hits the relevant endpoint. ONE shared inference client for both caption
    # (OpenAICaptioner) and text — the single llama-server/vLLM gateway (§5.1).
    from inference.client import OpenAICompatClient
    from modelsvc.residency import Residency

    inf = OpenAICompatClient(settings.inference_base_url or "")
    # ONE residency manager per service instance (§8.1), used by the only heavy
    # in-process model left — SigLIP. Since plan 16 removed the in-process caption
    # VLM, nothing contends with SigLIP for the GPU, so residency is always
    # non-exclusive (an idempotent ensure-loaded; no eviction, no preemption).
    residency = Residency(exclusive=False)
    embed = SiglipBackend(settings.embed_model_name, settings.embed_device, residency=residency)
    caption = build_caption_backend(settings, inf)
    # Caption-meaning text embeddings (§9) run in-process here on mac/jetson
    # (llama-server has no embedding backend); cloud (vLLM) keeps the OpenAI
    # `/embeddings` path (text_embed_model=None → client fallback).
    text_embed_model = None if settings.profile == "cloud" else settings.text_embed_model
    return CompositeBackend(
        embed=embed,
        caption=caption,
        text=TextBackend(
            inf, text_embed_model=text_embed_model, device=settings.embed_device,
            model_name=settings.planner_model,
        ),
        residency=residency,
    )


def build_caption_backend(settings, inf=None) -> CaptionBackend:
    """Config → `CaptionBackend` (design §4, §5.1). One captioner everywhere since
    plan 16: `OpenAICaptioner` over the OpenAI-compatible backend (llama-server on
    mac/jetson, vLLM on cloud).

    `inf` is the shared inference client `build_backend` also hands to
    `TextBackend` — the captioner reuses it instead of opening a second client.
    Falls back to building its own when called standalone (e.g. from tests) with
    no `inf` given. `use_fake_inference` swaps in a `FakeInferenceClient`. Imports
    are local so importing this module never pulls in the HTTP client until a
    caller actually needs it.
    """
    from captioning.openai_captioner import OpenAICaptioner

    if settings.use_fake_inference:
        from inference.fakes import FakeInferenceClient

        return CaptionBackend(OpenAICaptioner(FakeInferenceClient([]), "fake"))
    if inf is None:
        from inference.client import OpenAICompatClient

        inf = OpenAICompatClient(settings.inference_base_url or "")
    return CaptionBackend(OpenAICaptioner(inf, settings.caption_model or "fake"))
