"""Config -> `ModelBackend` factory (design §5.1, plan 15 task 2/3).

`build_backend` is the profile-config entrypoint to a running `models`
service: the real service main calls
`create_models_app(build_backend(settings))`.

Ruling R1: `FakeBackend` iff `settings.use_fake_embedder` — that flag alone
picks the fake path, kept deliberately simple. Per-concern fake selection
(e.g. a separate `use_fake_inference` for caption/text) is Tasks 3/4's
concern once those sub-backends exist.
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
    # caption backend's Captioner are both lazy), so this branch only pulls
    # torch/transformers when a request actually hits the relevant endpoint.
    # ONE shared inference client for both caption (mac/cloud OllamaCaptioner)
    # and text — the single Ollama gateway (design §5.1, plan 15 task 4).
    from inference.client import OpenAICompatClient
    from modelsvc.residency import Residency

    inf = OpenAICompatClient(settings.inference_base_url or "")
    # ONE residency manager per service instance (§8.1, plan 15 task 5),
    # shared by both heavy in-process sub-backends. Exclusive (evict +
    # preempt) only on jetson, where SigLIP and the caption VLM share one
    # CUDA device; mac/cloud stay non-exclusive (ruling R2 — behavior
    # unchanged, config-driven, not a platform branch).
    residency = Residency(exclusive=(settings.caption_backend == "inprocess"))
    embed = SiglipBackend(settings.embed_model_name, settings.embed_device, residency=residency)
    caption = build_caption_backend(settings, inf, residency=residency)
    return CompositeBackend(
        embed=embed,
        caption=caption,
        text=TextBackend(inf),
        residency=residency,
    )


def build_caption_backend(settings, inf=None, residency=None) -> CaptionBackend:
    """Profile → `CaptionBackend` (design §4, §5.1, plan 15 task 3) — the
    selection logic moved out of `config.Settings.build_captioner` (which now
    only feeds the soon-dead coordinator, Task 7) into the service that is
    actually allowed to load a captioner.

    `inf` is the shared inference client `build_backend` also hands to
    `TextBackend` (task 4) — the OllamaCaptioner path reuses it instead of
    opening a second Ollama client. Falls back to building its own when called
    standalone (e.g. from tests) with no `inf` given.

    `residency` (plan 15 task 5, §8.1) is injected straight into the
    `CaptionBackend` it returns; `None` (the default, e.g. when called
    directly from tests without a `build_backend`) means no residency
    wrapping at all — construction-only callers are unaffected.

    `use_fake_inference` always forces `OllamaCaptioner` over a
    `FakeInferenceClient`, on every profile — the offline test path stays
    reachable on jetson too. Imports are local so importing this module never
    pulls torch/transformers until a caller actually needs the jetson adapter.
    """
    if settings.use_fake_inference:
        from captioning.ollama_adapter import OllamaCaptioner
        from inference.fakes import FakeInferenceClient

        return CaptionBackend(OllamaCaptioner(FakeInferenceClient([]), "fake"), residency=residency)
    if settings.caption_backend == "inprocess":
        from captioning.vlm_adapter import VLMCaptioner

        return CaptionBackend(
            VLMCaptioner(settings.caption_model_id, device=settings.embed_device or "cuda"),
            residency=residency,
        )
    from captioning.ollama_adapter import OllamaCaptioner

    if inf is None:
        from inference.client import OpenAICompatClient

        inf = OpenAICompatClient(settings.inference_base_url or "")
    return CaptionBackend(OllamaCaptioner(inf, settings.caption_model or "fake"), residency=residency)
