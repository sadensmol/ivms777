"""In-process caption-meaning text embedder (design §4, §9).

Since plan 16 dropped Ollama, the dedicated text embedder (`nomic-embed-text-v1.5`
by default) no longer has a server — llama-server holds only the one gemma GGUF.
So it loads IN-PROCESS in the `models` service (the single model process, §5.1),
next to SigLIP, and `TextBackend.text_embed` uses it instead of an HTTP backend.

Only the `models` service imports this module (via `TextBackend`); `torch` /
`transformers` are imported LAZILY inside `_load`, so importing
`embedding.text_embedder` — or `modelsvc.backends` — stays torch-free for
`app`/`worker` and the test suite. The required `search_query:` /
`search_document:` task prefixes are applied by the caller
(`embedding/caption_text.py`), so this module just encodes + mean-pools +
L2-normalises.
"""

from functools import lru_cache


class TextEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings from a HF encoder
    (`nomic-embed-text-v1.5` by default) on `device` (cpu on mac, cuda on
    jetson). Loaded once; `transformers`/`torch` imported lazily on first use."""

    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401 - imported for side effect + used below
        from transformers import AutoModel, AutoTokenizer

        # nomic's encoder ships custom modeling code (needs `einops`); trust it.
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True
        ).to(self._device).eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._load()
        tokens = self._tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._model(**tokens)
        # Mean-pool the token embeddings over the attention mask, then L2-normalise.
        hidden = output.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return pooled.cpu().tolist()


@lru_cache
def get_text_embedder(model_name: str, device: str) -> TextEmbedder:
    """Cached so the encoder loads once per (model, device) and is reused across
    requests — mirrors `embedding.siglip.get_siglip_embedder`."""
    return TextEmbedder(model_name, device)
