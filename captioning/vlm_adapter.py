"""In-process transformers 4-bit vision captioner (Jetson) (design §3.1, §4, §8.1).

Ollama's CUDA build runs vision on CPU on JP7, so the Jetson captions through
this in-process `transformers` adapter instead. The heavy `torch`/`transformers`
load is board-only and lazy — `_real_loader` imports them inside the function so
this module (and the test suite) import cleanly without torch. Loaded and freed
by the model coordinator under the INGEST_CAPTION lease (§8.1).
"""

import io
import json
from collections.abc import Callable

from PIL import Image

from captioning.base import CaptionResult
from inference.prompts import _DEFAULT_SYSTEM, _user_text


def _real_loader(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map=device, dtype=torch.float16).eval()
    return model, proc


class VLMCaptioner:
    """In-process vision captioner (Jetson): Qwen2.5-VL 4-bit via transformers on the
    cu132 GPU — Ollama's CUDA build runs vision on CPU on JP7 (design §3.1, §4). Loaded
    and freed by the model coordinator under the INGEST_CAPTION lease (§8.1)."""

    def __init__(self, model_id: str, *, device: str = "cuda", footprint_mb: int = 2700,
                 _loader: Callable = _real_loader):
        self._model_id = model_id
        self._device = device
        self._footprint = footprint_mb
        self._loader = _loader
        self._model = None
        self._proc = None
        self.name = f"{model_id} (in-process 4-bit)"
        self.caption_model = "qwen2.5-vl-3b-inprocess"

    def load(self) -> None:
        if self._model is None:
            self._model, self._proc = self._loader(self._model_id, self._device)

    def release(self) -> None:
        self._model = None
        self._proc = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001,S110 - release is best-effort; it must never raise
            pass

    def footprint_mb(self) -> int:
        return self._footprint

    def caption(self, image: bytes, dimensions, *, should_preempt=lambda: False) -> CaptionResult:
        if self._model is None:
            raise RuntimeError("VLMCaptioner.caption called before load()")
        img = Image.open(io.BytesIO(image)).convert("RGB")
        msgs = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _user_text(dimensions)}]},
        ]
        prompt = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[prompt], images=[img], return_tensors="pt").to(self._device)
        out = self._model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                   stopping_criteria=self._stop(should_preempt))
        n = len(inputs["input_ids"][0])
        text = self._proc.batch_decode([out[0][n:]], skip_special_tokens=True)[0]
        obj = _extract_json(text)
        caption = (obj.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"VLM returned no caption: {text[:120]!r}")
        return CaptionResult(caption, obj.get("title") or "",
                             obj.get("description") or "", obj.get("tags") or {})

    def _stop(self, should_preempt):
        # A StoppingCriteria that aborts generation when an interactive workload preempts.
        from transformers import StoppingCriteria, StoppingCriteriaList

        from inference.client import InferenceCancelled

        model_id = self._model_id
        class _Preempt(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                if should_preempt():
                    raise InferenceCancelled(model_id)  # mapped to Preempted by the stage
                return False
        return StoppingCriteriaList([_Preempt()])


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in VLM output: {text[:120]!r}")
    return json.loads(text[start:end + 1])
