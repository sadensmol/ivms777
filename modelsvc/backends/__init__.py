"""Config -> `ModelBackend` factory (design §5.1).

`build_backend` is the profile-config entrypoint to a running `models`
service: the real service main calls
`create_models_app(build_backend(settings))`.

Ruling R1: `FakeBackend` iff `settings.use_fake_embedder` — that flag alone
picks the fake path, kept deliberately simple.

Since plan 21 the four residency units are built by `SlotManager` from the
selected **catalog entries** (design §4.1), not from hardcoded model names. The
service holds no database, so it starts on the profile defaults; `app` pushes the
user's stored selection over `PUT /models/slots`.
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
    # Board-only, but torch-free to CONSTRUCT: every unit registers lazy-importing
    # load/free callables, so this branch pulls torch/transformers (or spawns
    # llama-server) only when a request actually loads a model. ONE shared
    # inference client for both caption and text — the single gateway (§5.1).
    import psutil

    from inference.client import OpenAICompatClient
    from models import slots as slot_defaults
    from modelsvc.downloads import Downloads
    from modelsvc.governor import MemoryGovernor
    from modelsvc.llm_process import build_llm_process
    from modelsvc.registry import ModelRegistry
    from modelsvc.scheduler import Scheduler
    from modelsvc.slots import SlotManager

    inf = OpenAICompatClient(settings.inference_base_url or "")
    registry = ModelRegistry()
    manager = SlotManager(
        settings,
        registry,
        llm_factory=lambda planner, caption: build_llm_process(
            settings, planner=planner, caption=caption
        ),
    )
    # The service has no DB (§4.1): it starts on the profile defaults and `app`
    # pushes the stored selection on its next resources poll.
    manager.apply(slot_defaults.resolve_keys(None, settings))

    governor = MemoryGovernor(
        registry,
        measure_free_mb=lambda: psutil.virtual_memory().available / (1024 * 1024),
        budget_mb=settings.ram_budget_mb,
    )
    scheduler = Scheduler(
        governor, concurrency=settings.gpu_concurrency, idle_ttl_s=settings.llm_idle_ttl_s
    )

    # cloud keeps the OpenAI `/embeddings` path for caption text and hosts no
    # in-process text embedder, so its text unit is never registered.
    text_worker = (
        None if settings.profile == "cloud" else (lambda: manager.worker("text_embed"))
    )
    embed = SiglipBackend(lambda: manager.worker("image_embed"))
    caption = build_caption_backend(settings, inf, entry=manager.entry("caption"))
    return CompositeBackend(
        embed=embed,
        caption=caption,
        text=TextBackend(
            inf, text_worker=text_worker, model_name=manager.entry("planner").key
        ),
        registry=registry,
        governor=governor,
        scheduler=scheduler,
        text_embed_needs=() if text_worker is None else ("text_embed",),
        slots=manager,
        downloads=Downloads(settings.data_dir / "models"),
        profile=settings.profile,
    )


def build_caption_backend(settings, inf=None, *, entry=None) -> CaptionBackend:
    """Config → `CaptionBackend` (design §4, §5.1). One captioner everywhere since
    plan 16: `OpenAICaptioner` over the OpenAI-compatible backend (llama-server on
    mac/jetson, vLLM on cloud).

    `inf` is the shared inference client `build_backend` also hands to
    `TextBackend` — the captioner reuses it instead of opening a second client.
    Falls back to building its own when called standalone (e.g. from tests) with
    no `inf` given. `entry` is the selected `caption` catalog entry (§4.1): its key
    is the model name stored with every caption, its `prompt_template` selects the
    prompt. `use_fake_inference` swaps in a `FakeInferenceClient`. Imports are local
    so importing this module never pulls in the HTTP client until a caller actually
    needs it.
    """
    from captioning.openai_captioner import OpenAICaptioner

    if settings.use_fake_inference:
        from inference.fakes import FakeInferenceClient

        return CaptionBackend(OpenAICaptioner(FakeInferenceClient([]), "fake"))
    if inf is None:
        from inference.client import OpenAICompatClient

        inf = OpenAICompatClient(settings.inference_base_url or "")
    if entry is None:
        from models import catalog

        entry = catalog.get("caption", catalog.default_key("caption", settings.profile))
    return CaptionBackend(
        OpenAICaptioner(inf, entry.key, prompt_key=entry.prompt_template)
    )
