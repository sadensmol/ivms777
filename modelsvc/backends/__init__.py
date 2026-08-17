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
    # Board-only, but torch-free to CONSTRUCT: SigLIP/nomic/gemma all register
    # lazy-importing load/free callables, so this branch pulls torch/transformers
    # (or spawns llama-server) only when a request actually loads a model. ONE
    # shared inference client for both caption and text — the single gateway (§5.1).
    import psutil

    from inference.client import OpenAICompatClient
    from modelsvc.governor import MemoryGovernor
    from modelsvc.llm_process import build_llm_process
    from modelsvc.registry import ModelRegistry, ModelSpec
    from modelsvc.scheduler import Scheduler
    from modelsvc.torch_process import TorchWorker

    inf = OpenAICompatClient(settings.inference_base_url or "")
    llm = build_llm_process(settings)
    costs = settings.model_cost_mb

    # Caption-meaning text embeddings (§9) run in-process on mac/jetson (nomic);
    # cloud (vLLM) keeps the OpenAI `/embeddings` path (text_embed_model=None).
    text_embed_model = None if settings.profile == "cloud" else settings.text_embed_model

    registry = ModelRegistry()

    # SigLIP and nomic run in KILLABLE children (plan 20, §8.1): freeing a torch
    # model in-process leaves ~2.7 GB of CUDA-context RSS behind — precisely the RAM
    # gemma then cannot have — so `free` is a process kill, and `alive` lets the
    # registry respawn a child that died on its own.
    siglip_worker = TorchWorker(
        "embedding.siglip:SiglipEmbedder", (settings.embed_model_name, settings.embed_device)
    )
    registry.register(
        ModelSpec(
            "siglip",
            siglip_worker.start,
            siglip_worker.stop,
            costs["siglip"],
            alive=siglip_worker.is_alive,
        )
    )

    text_worker = None
    if text_embed_model is not None:
        text_worker = TorchWorker(
            "embedding.text_embedder:TextEmbedder",
            (text_embed_model, settings.embed_device),
            warm="warm",  # this encoder loads lazily; resident must mean resident
        )
        registry.register(
            ModelSpec(
                "nomic",
                text_worker.start,
                text_worker.stop,
                costs["nomic"],
                alive=text_worker.is_alive,
            )
        )

    # gemma is registered TWICE (design §3.1/§8.1): `gemma` is text-only (chat,
    # planner) and `gemma-vision` adds the ~531 MB projector for captioning. They are
    # ONE llama-server child on ONE port, so they are mutually exclusive — loading
    # either evicts the other, keeping the registry's resident set honest.
    def _load_gemma(vision: bool) -> None:
        sibling = "gemma" if vision else "gemma-vision"
        registry.unload(sibling)  # no-op unless the other mode is resident
        llm.load(vision=vision)

    # `alive` is the child's real state: llama-server aborts on its own when a CUDA
    # malloc fails (an image decode on the 8 GB board), and without this probe the
    # registry keeps claiming gemma is resident and every request hits a dead port
    # (§8.1). SigLIP/nomic need no probe — an in-process model cannot die alone.
    registry.register(
        ModelSpec(
            "gemma", lambda: _load_gemma(False), llm.free, costs["gemma"], alive=llm.is_loaded
        )
    )
    registry.register(
        ModelSpec(
            "gemma-vision",
            lambda: _load_gemma(True),
            llm.free,
            costs.get("gemma-vision", costs["gemma"]),
            alive=llm.is_loaded,
        )
    )

    governor = MemoryGovernor(
        registry,
        measure_free_mb=lambda: psutil.virtual_memory().available / (1024 * 1024),
        budget_mb=settings.ram_budget_mb,
    )
    scheduler = Scheduler(
        governor, concurrency=settings.gpu_concurrency, idle_ttl_s=settings.llm_idle_ttl_s
    )

    embed = SiglipBackend(siglip_worker)
    caption = build_caption_backend(settings, inf)
    return CompositeBackend(
        embed=embed,
        caption=caption,
        text=TextBackend(inf, text_worker=text_worker, model_name=settings.planner_model),
        registry=registry,
        governor=governor,
        scheduler=scheduler,
        text_embed_needs=() if text_embed_model is None else ("nomic",),
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
