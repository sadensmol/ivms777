"""Caption-meaning text embeddings (§9) — a dedicated **text** embedder, NOT SigLIP.

SigLIP's text tower is trained for image↔text, so text↔text cosines collapse into a
narrow band with no separation (measured; design §4). A purpose-built text embedder
(`nomic-embed-text-v1.5` by default — benchmark in §4) gives real neighbour ordering,
so a query can KNN meaning-similar captions out of a huge library without the LLM
looping. Since plan 16 dropped Ollama, it is loaded IN-PROCESS in the `models`
service (`embedding/text_embedder.py`); `client.embed` reaches it over the models
gateway (`/text/embed`), same call shape as before.

Such embedders need a task prefix on each input; the pair is model-specific, so it
lives here (one place) keyed by model name. Query and document use different prefixes.
Consumed as **top-k KNN**, never a fixed cosine floor (nomic's baseline is high).
"""


def _prefix(model: str, *, is_query: bool) -> str:
    """The task prefix a text embedder wants, by model family. Empty for models that
    take none (or an unknown model — then it is a plain cosine, still usable)."""
    if model.startswith("nomic"):
        return "search_query: " if is_query else "search_document: "
    if model.startswith("mxbai") and is_query:
        return "Represent this sentence for searching relevant passages: "
    return ""


def embed_caption_texts(client, model: str, texts: list[str], *, is_query: bool) -> list[list[float]]:
    """Embed caption text (documents) or a search query with `model` via `client`
    (the OpenAI-compatible `/embeddings` endpoint), applying the model's task prefix.
    `client` is an `inference.client.InferenceClient` (or a fake in tests)."""
    prefix = _prefix(model, is_query=is_query)
    return client.embed(model, [prefix + t for t in texts])
