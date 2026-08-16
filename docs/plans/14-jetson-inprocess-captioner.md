# Jetson In-Process VLM Captioner (caption adapters) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make captioning work on the Jetson GPU by running the vision model **in-process via `transformers`/cu132 torch** (Ollama's CUDA build runs vision on CPU on JP7). Do it behind a **captioner adapter** so Mac/cloud keep the exact Ollama path and the Jetson uses the in-process VLM — selected by profile, with the model coordinator managing whichever adapter is active.

**Proven on the board (smoke test):** `Qwen/Qwen2.5-VL-3B-Instruct` 4-bit (bitsandbytes) loads on the Orin GPU via our cu132 torch — **no NVML crash** (the wall that killed vLLM), **2.67 GB** resident, ~12.5 s/caption, needs `gcc` (Triton JIT) + `bitsandbytes` + `accelerate` in the image.

**Architecture:** A `captioning.Captioner` Protocol with two adapters — `OllamaCaptioner` (HTTP, mac/cloud, wraps today's exact behavior) and `VLMCaptioner` (in-process transformers 4-bit, jetson). `config.build_captioner()` picks by profile. The caption stage and the `ModelCoordinator` both use the adapter: the coordinator's `INGEST_CAPTION` workload holds a `CAPTIONER` resource whose `load()/release()/footprint_mb()` the adapter implements — so on Mac `load()` warms the Ollama model (unchanged) and on Jetson it loads the in-process VLM.

**Tech Stack:** Python 3.12, `uv`, `transformers` 5.15 + `bitsandbytes` + `accelerate` (jetson only), cu132 torch, Qwen2.5-VL, SQLite, the existing `ModelCoordinator` (plan 13).

**Spec:** `docs/design.md` §3.1 (deploy profiles), §4 (models), §8/§8.1 (ingest + coordinator). This plan updates all of them.

## Global Constraints

- **JETSON-ONLY behavior change. Mac and cloud must be byte-identical to today.** The mechanism: profile-selected adapter. Mac/cloud → `OllamaCaptioner` (today's code path). Jetson → `VLMCaptioner`. No shared-code change may alter the Ollama path's behavior.
- `docs/design.md` is the source of truth — update §3.1/§4/§8/§8.1 in the same work as the code.
- Python `>=3.12`, deps via `uv`. Tests in `tests/`, `uv run pytest`. Every new module needs tests.
- **No git commits** — the user commits. Implementers never `git add`/`git commit`.
- Tests use the `conn` fixture (`tests/conftest.py`) and `create_app(settings)` patterns; never raw `sqlite3.connect(":memory:")` (schema has vec0/fts5).
- The heavy real VLM load (`transformers` + bitsandbytes + a 7.5 GB download) is **board-only** — it must NOT run in the test suite. Tests exercise the adapter logic with fakes/monkeypatch; the real load is covered by the on-board verification task.
- Coordinator invariants from plan 13 hold (lease, piggyback, hard-preempt, unconditional release). This plan generalizes the coordinator's residency to "resources" without breaking them.

---

### Task 1: `captioning` package — base + `OllamaCaptioner` (mac/cloud path, unchanged behavior)

Extract today's caption behavior into an adapter, provably identical for the Ollama path.

**Files:**
- Create: `captioning/__init__.py` (empty)
- Create: `captioning/base.py`
- Create: `captioning/ollama_adapter.py`
- Test: `tests/test_captioner_ollama.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class CaptionResult: caption:str; title:str; description:str; tags:dict[str,list[str]]`
  - `class Captioner(Protocol)`:
    - `def caption(self, image: bytes, dimensions: list[str], *, should_preempt: Callable[[],bool]=lambda: False) -> CaptionResult`
    - `def load(self) -> None` — make the model resident (coordinator calls on lease enter)
    - `def release(self) -> None` — free it (coordinator calls on lease exit)
    - `def footprint_mb(self) -> int`
    - `name: str` — label for the coordinator model-set + resource bar (e.g. the ollama tag, or `"qwen2.5-vl-3b (in-process)"`)
    - `caption_model: str` — the value stored in `photos.caption_model`
  - `class OllamaCaptioner(Captioner)`: `__init__(self, client, model, footprint_mb=None)`; `caption()` = `caption_messages(model, encode_image(image), dimensions)` → `client.complete(model, msgs, json_schema=CAPTION_SCHEMA, should_stop=should_preempt)` → parse JSON → `CaptionResult`; on `InferenceCancelled` re-raise (the stage maps it to `Preempted`); `load()` = `client.warm(model)`; `release()` = `client.evict(model)`; `footprint_mb()` from `workloads.FOOTPRINT_MB` (fallback `_FALLBACK_LLM_MB`); `name`/`caption_model` = `model`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_captioner_ollama.py
import json
from captioning.ollama_adapter import OllamaCaptioner
from captioning.base import CaptionResult


class FakeClient:
    def __init__(self, payload): self._p = payload; self.warmed=[]; self.evicted=[]
    def complete(self, model, messages, *, json_schema=None, should_stop=None, timeout=120.0):
        return json.dumps(self._p)
    def warm(self, model, *, timeout=120.0): self.warmed.append(model)
    def evict(self, model, *, timeout=30.0): self.evicted.append(model)


def test_caption_parses_into_result():
    payload = {"caption":"a dog","title":"Dog","description":"a brown dog","tags":{"subject":["dog"]}}
    cap = OllamaCaptioner(FakeClient(payload), "qwen2.5vl:7b")
    r = cap.caption(b"imgbytes", ["subject","scene"])
    assert isinstance(r, CaptionResult)
    assert r.caption=="a dog" and r.title=="Dog" and r.tags=={"subject":["dog"]}
    assert cap.caption_model=="qwen2.5vl:7b" and cap.name=="qwen2.5vl:7b"


def test_load_release_warm_evict_the_model():
    c = FakeClient({"caption":"x","title":"x","description":"x","tags":{}})
    cap = OllamaCaptioner(c, "qwen2.5vl:7b")
    cap.load(); cap.release()
    assert c.warmed==["qwen2.5vl:7b"] and c.evicted==["qwen2.5vl:7b"]
```

- [ ] **Step 2: Run — fails** (`uv run pytest tests/test_captioner_ollama.py -v` → import error).

- [ ] **Step 3: Implement `captioning/base.py`**

```python
# captioning/base.py
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    title: str
    description: str
    tags: dict[str, list[str]]


class Captioner(Protocol):
    name: str
    caption_model: str
    def caption(self, image: bytes, dimensions: list[str], *,
                should_preempt: Callable[[], bool] = lambda: False) -> CaptionResult: ...
    def load(self) -> None: ...
    def release(self) -> None: ...
    def footprint_mb(self) -> int: ...
```

- [ ] **Step 4: Implement `captioning/ollama_adapter.py`**

```python
# captioning/ollama_adapter.py
import json
from collections.abc import Callable

from captioning.base import CaptionResult
from inference.client import encode_image
from inference.prompts import CAPTION_SCHEMA, caption_messages
from models import workloads as wl


class OllamaCaptioner:
    """Captioning over the OpenAI-compatible inference backend (Ollama on mac/cloud).
    Exactly today's caption path, wrapped as an adapter (design §4)."""

    def __init__(self, client, model: str, footprint_mb: int | None = None):
        self._client = client
        self.name = model
        self.caption_model = model
        self._footprint = footprint_mb

    def caption(self, image: bytes, dimensions, *, should_preempt=lambda: False) -> CaptionResult:
        msgs = caption_messages(self.caption_model, encode_image(image), dimensions)
        raw = self._client.complete(
            self.caption_model, msgs, json_schema=CAPTION_SCHEMA, should_stop=should_preempt
        )  # InferenceCancelled propagates; the caption stage maps it to Preempted
        obj = json.loads(raw)
        return CaptionResult(obj["caption"], obj["title"], obj["description"], obj.get("tags") or {})

    def load(self) -> None:
        self._client.warm(self.caption_model)

    def release(self) -> None:
        self._client.evict(self.caption_model)

    def footprint_mb(self) -> int:
        if self._footprint is not None:
            return self._footprint
        return wl.FOOTPRINT_MB.get(self.caption_model, wl._FALLBACK_LLM_MB)
```

- [ ] **Step 5: Run — passes.** `uv run pytest tests/test_captioner_ollama.py -v` + `uv run pytest -q` (no regressions).

---

### Task 2: `VLMCaptioner` — in-process transformers 4-bit adapter (Jetson)

The in-process adapter. Its heavy `transformers` load is **lazy and board-only**; tests inject fakes.

**Files:**
- Create: `captioning/vlm_adapter.py`
- Test: `tests/test_captioner_vlm.py`

**Interfaces:**
- Consumes: `captioning.base`, `inference.prompts._DEFAULT_SYSTEM` + `_user_text` (reuse the SAME prompt text as Ollama), `models.workloads`.
- Produces: `class VLMCaptioner(Captioner)`: `__init__(self, model_id, *, device="cuda", footprint_mb=2700, _loader=None)`. `_loader` is an injectable callable returning `(model, processor)` — defaults to the real transformers loader; tests pass a fake so no torch/download runs. `caption()` builds a transformers chat (system=`_DEFAULT_SYSTEM`, user=`_user_text(dimensions)` + image), runs `model.generate` with a `should_preempt` stopping hook, extracts the JSON object from the decoded text (tolerant: find first `{`…last `}`), returns `CaptionResult`. `load()` calls `_loader` once (idempotent). `release()` drops refs + `torch.cuda.empty_cache()`. `caption_model` = a stable string (e.g. `"qwen2.5-vl-3b-inprocess"`) written to `photos.caption_model`.

- [ ] **Step 1: Write failing tests** (fake loader — NO real model)

```python
# tests/test_captioner_vlm.py
import pytest
from captioning.vlm_adapter import VLMCaptioner
from captioning.base import CaptionResult


class FakeProcessor:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True): return "PROMPT"
    def __call__(self, text=None, images=None, return_tensors=None):
        class B(dict):
            def to(self, dev): return self
        b = B(); b["input_ids"] = [[0,1,2]]; return b
    def batch_decode(self, ids, skip_special_tokens=True):
        return ['Sure: {"caption":"a cat","title":"Cat","description":"a black cat","tags":{"subject":["cat"]}}']


class FakeModel:
    def generate(self, **kw): return [[0,1,2,3,4,5]]
    def eval(self): return self


def _fake_loader(model_id, device):
    return FakeModel().eval(), FakeProcessor()


def test_vlm_caption_extracts_json_result():
    cap = VLMCaptioner("Qwen/Qwen2.5-VL-3B-Instruct", _loader=_fake_loader)
    cap.load()
    r = cap.caption(b"imgbytes", ["subject"])
    assert isinstance(r, CaptionResult)
    assert r.caption == "a cat" and r.tags == {"subject": ["cat"]}
    assert cap.footprint_mb() == 2700
    cap.release()   # must not raise even without CUDA


def test_caption_before_load_raises():
    cap = VLMCaptioner("m", _loader=_fake_loader)
    with pytest.raises(RuntimeError):
        cap.caption(b"x", ["subject"])
```

- [ ] **Step 2: Run — fails.**

- [ ] **Step 3: Implement `captioning/vlm_adapter.py`**

```python
# captioning/vlm_adapter.py
import json
from collections.abc import Callable

from PIL import Image
import io

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
        except Exception:  # noqa: BLE001 - release is best-effort
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
        # Strict on the caption: the VLM decode is UNCONSTRAINED (no json_schema like
        # Ollama), so a missing caption must RAISE → the stage retries the job rather
        # than writing an empty half-row (ingest/caption.py invariant). title/desc/tags
        # stay tolerant — they're enhancements.
        caption = (obj.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"VLM returned no caption: {text[:120]!r}")
        return CaptionResult(caption, obj.get("title") or "",
                             obj.get("description") or "", obj.get("tags") or {})

    def _stop(self, should_preempt):
        # A StoppingCriteria that aborts generation when an interactive workload preempts.
        from inference.client import InferenceCancelled
        from transformers import StoppingCriteria, StoppingCriteriaList

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
```

> Note: `_extract_json` mirrors the tolerant parsing chat already uses (`chat/agent.py::_turn`). The `StoppingCriteria` raising `InferenceCancelled` matches the Ollama adapter's cancellation contract so the caption stage handles both identically.

- [ ] **Step 4: Run — passes** (fake loader; no torch needed for the JSON-path test). Guard: if `import torch` isn't available in the test env, `release()` still must not raise (the `except` covers it) and the JSON test never imports torch.

---

### Task 3: `config.build_captioner()` + profile selection

**Files:**
- Modify: `config.py` (`PROFILE_DEFAULTS` add `caption_backend`; add `caption_model_id`; add `build_captioner()`)
- Test: `tests/test_captioner_config.py`

**Interfaces:**
- Produces: `Settings.caption_backend: Literal["ollama","inprocess"] | None`, `Settings.caption_model_id: str | None` (HF repo for inprocess), `Settings.build_captioner(self, client) -> Captioner`.
- `PROFILE_DEFAULTS`: mac/cloud → `caption_backend="ollama"`; jetson → `caption_backend="inprocess"`, `caption_model_id="Qwen/Qwen2.5-VL-3B-Instruct"`. (Follow the `int | None = None` field pattern so `_apply_profile_defaults` fills them.)
- `build_captioner(client)`: `"inprocess"` → `VLMCaptioner(self.caption_model_id, device=self.embed_device or "cuda")`; else → `OllamaCaptioner(client, self.caption_model or "fake")`. `use_fake_inference` → `OllamaCaptioner(FakeInferenceClient(...), "fake")` (keeps the offline path on any profile).

- [ ] **Step 1: failing test**

```python
# tests/test_captioner_config.py
from config import Settings
from captioning.ollama_adapter import OllamaCaptioner
from captioning.vlm_adapter import VLMCaptioner
from inference.fakes import FakeInferenceClient


def test_mac_profile_uses_ollama_captioner(tmp_path):
    s = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    assert s.caption_backend == "ollama"
    assert isinstance(s.build_captioner(FakeInferenceClient([])), OllamaCaptioner)


def test_jetson_profile_uses_inprocess_captioner(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True)
    assert s.caption_backend == "inprocess"
    assert s.caption_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    cap = s.build_captioner(FakeInferenceClient([]))
    assert isinstance(cap, VLMCaptioner)   # NOT constructed/loaded — just built
```

- [ ] **Step 2-4:** implement the config changes (fields default `None`; add to each `PROFILE_DEFAULTS` entry; `build_captioner` as above), run tests, `uv run pytest -q` no regressions.

> Guard for tests: `build_captioner` for `"inprocess"` only *constructs* `VLMCaptioner` (cheap) — it must NOT call `.load()` (which would import torch/download). Loading happens only under the coordinator lease on the board.

---

### Task 4: Refactor the caption stage to use the adapter

**Files:**
- Modify: `ingest/caption.py` (`caption_handler` takes a `Captioner` + the embed client; drop the direct `client.complete` caption call — keep `_embed_caption` on the embed client)
- Modify: `ingest/pipeline.py` (build the captioner via `settings.build_captioner(client)`; pass it to `caption_handler` and to the coordinator — Task 5)
- Test: `tests/test_caption_stage.py` (extend/adjust existing caption tests)

**Interfaces:**
- `caption_handler(derived, captioner, embed_client, embed_model, dimensions, detail_px, should_preempt=...) -> StageHandler`.
- The handler: read image bytes → `result = captioner.caption(image, dimensions, should_preempt=should_preempt)` (catch `InferenceCancelled` → `raise Preempted`); `UPDATE photos SET caption=?, caption_model=?, ai_title=?, ai_description=?` using `result.*` and `captioner.caption_model`; `_write_vlm_tags(conn, photo_id, result.tags)`; `_embed_caption(conn, embed_client, embed_model, photo_id, result.caption)`; `reindex_fts`.

- [ ] **Step 1: failing test** — a fake captioner returns a fixed `CaptionResult`; assert the row is written (caption/title/description/tags) and a preempting captioner (raises `InferenceCancelled`) makes the handler raise `Preempted` and leaves the job pending. Reuse the `conn` fixture + `tests/factories.add_photo` + a stored thumbnail (see the existing caption test for the thumbnail-write helper).

- [ ] **Step 2-4:** implement; in `ingest/pipeline.py` group 2b, replace `caption_handler(context.derived, client, caption_model, ...)` with `caption_handler(context.derived, captioner, client, settings.caption_embed_model, list(vocab.dimensions), settings.thumb_detail_px, should_preempt=should_preempt)` where `captioner = settings.build_captioner(client)` (built once per pass). Keep `backfill_caption_vectors(conn, client, ...)` on the Ollama client (unchanged — §9 embeddings stay on Ollama, jetson included). Run the full suite; **verify the existing mac-style caption tests still pass with the OllamaCaptioner path** (byte-identical behavior).

---

### Task 5: Coordinator manages the active captioner adapter (generalize residency to resources)

Make `INGEST_CAPTION` hold a `CAPTIONER` resource whose load/release/footprint the injected adapter implements. On mac the adapter is `OllamaCaptioner`, so `load()` = warm the Ollama model — **identical to today**; on jetson it loads the in-process VLM.

**Files:**
- Modify: `models/workloads.py` (`CAPTIONER` sentinel; `model_set("INGEST_CAPTION") -> {CAPTIONER}`)
- Modify: `models/coordinator.py` (accept an injected `captioner`; resolve each want-member to `(load, release, footprint)`; `CAPTIONER` → the adapter; `SIGLIP` → siglip funcs; else → `client.warm/evict`; RAM guard uses per-member footprint incl. `captioner.footprint_mb()`)
- Modify: `web/deps.py` (`make_coordinator` builds the captioner via `settings.build_captioner(client)`, injects it into the `ModelCoordinator`, and **exposes it as `coordinator.captioner`**)
- Modify: `ingest/pipeline.py` + `ingest/cli.py` (**SHARED INSTANCE** — see below)
- Modify: `models/resources.py` (resource bar: show the captioner's `name` for `INGEST_CAPTION`)
- Test: `tests/test_model_coordinator.py` (add: INGEST_CAPTION loads/releases the injected captioner; mac-style OllamaCaptioner load = warm)

**⚠️ SHARED INSTANCE (critical):** the coordinator LOADS the captioner on `require("INGEST_CAPTION")` enter, and the caption STAGE (`drain_pass` group 2b, inside that lease) then calls `captioner.caption(...)`. These MUST be the **same instance** — otherwise the coordinator loads instance A onto the GPU while the stage calls an unloaded instance B (`VLMCaptioner.caption` raises "before load()"). So:
- `drain_pass` group 2b (Task 4 currently builds its own `settings.build_captioner(client)`) must instead use `coordinator.captioner` when a coordinator is present: `captioner = coordinator.captioner if coordinator is not None else settings.build_captioner(client)`.
- The worker (`ingest/cli.py`) already builds the coordinator via `make_coordinator` — no separate captioner build; the stage pulls it off the coordinator.
- **App inline drain path (`coordinator=None`):** `drain_pass` builds its own captioner (best-effort). `OllamaCaptioner` needs no explicit load (Ollama auto-loads on the `complete` call) so mac's inline drain is unaffected. For an in-process `VLMCaptioner` with no coordinator, `caption()` will raise "before load()" → the drain marks the job failed and the WORKER (which has a coordinator) re-runs it — acceptable best-effort degradation; the worker is the real caption path on jetson. Document this in a comment; do NOT lazy-load the VLM outside a lease (it would bypass the RAM budget).

**Interfaces:**
- `models/workloads.py`: `CAPTIONER = "captioner"`. `model_set("INGEST_CAPTION") -> frozenset({CAPTIONER})`. Keep `FOOTPRINT_MB` for SIGLIP/LLMs; `CAPTIONER`'s footprint is resolved by the coordinator via the adapter (not the static table).
- `ModelCoordinator.__init__(..., captioner=None)`. Add `_resolve(member) -> (load: Callable, release: Callable, footprint_mb: int)`:
  - `member == wl.SIGLIP` → `(self._do_load_siglip, self._release_siglip, wl.FOOTPRINT_MB[wl.SIGLIP])`
  - `member == wl.CAPTIONER` → `(self._captioner.load, self._captioner.release, self._captioner.footprint_mb())` (raise if `captioner is None`)
  - else (LLM tag) → `(lambda: self._client.warm(member), lambda: self._client.evict(member), wl.FOOTPRINT_MB.get(member, wl._FALLBACK_LLM_MB))`
  - RAM guard: `fits = sum(footprint of each member) <= budget`. `_reconcile`/`_release` call the resolved load/release (the `_release` try/except robustness stays).

- [ ] **Step 1: failing tests**

```python
# add to tests/test_model_coordinator.py
class FakeCaptioner:
    def __init__(self): self.loaded=0; self.released=0
    name="cap"; caption_model="cap"
    def load(self): self.loaded+=1
    def release(self): self.released+=1
    def footprint_mb(self): return 2700

def test_ingest_caption_loads_and_releases_the_captioner(conn):
    client = FakeClient(); cap = FakeCaptioner()
    coord = ModelCoordinator(conn, client, holder="worker", budget_mb=99_999,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: None, release_siglip=lambda: None, captioner=cap, sleep=lambda s: None)
    with coord.require("INGEST_CAPTION"):
        assert cap.loaded == 1
        assert client.warmed == []           # in-process path: NO ollama warm
    assert cap.released == 1
```

(Also update any existing INGEST_CAPTION test that assumed `client.warm("qwen2.5vl:3b")`: with the adapter, the *captioner* is loaded, not a raw LLM warm — for an OllamaCaptioner that in turn warms the model, but the coordinator now calls `captioner.load()`.)

- [ ] **Step 2-4:** implement `_resolve` + injection; `make_coordinator` builds+injects the captioner; resource bar shows the captioner name for INGEST_CAPTION. Run full suite; **assert mac path unchanged** (a coordinator built with an `OllamaCaptioner` warms/evicts the ollama model on INGEST_CAPTION, same net effect as before).

---

### Task 6: Jetson image deps + design.md

**Files:**
- Modify: `Dockerfile.jetson` (add `gcc`; add `bitsandbytes` + `accelerate` to the jetson install)
- Modify: `pyproject.toml` (a `jetson` optional-dependency group `bitsandbytes, accelerate`, OR install them in Dockerfile.jetson only — keep them OFF the mac/cloud image)
- Modify: `docs/design.md` §3.1, §4, §8, §8.1
- Modify: `captioning` — ensure `captioning` is in the Docker `COPY` list + `pyproject` `only-include` (same gap that broke `models/`)

- [ ] **Step 1:** `Dockerfile.jetson` — add before the python source copy:
  ```dockerfile
  RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
  ```
  and after `uv sync`, install the jetson caption deps into the venv (they must NOT be on mac):
  ```dockerfile
  RUN VIRTUAL_ENV=/app/.venv uv pip install bitsandbytes accelerate
  ```
  Add `COPY captioning ./captioning`.
- [ ] **Step 2:** `Dockerfile` (mac/cloud) — add `COPY captioning ./captioning` (the package must ship; it's imported on all profiles for the Ollama adapter). Do NOT add gcc/bitsandbytes there.
- [ ] **Step 3:** `pyproject.toml` — add `"captioning"` to `[tool.hatch.build.targets.wheel] only-include`.
- [ ] **Step 4:** `docs/design.md`:
  - §4 models table: caption row → "per profile: Ollama VLM (mac/cloud) OR in-process Qwen2.5-VL-3B 4-bit via transformers (jetson)".
  - §3.1: replace the "captions stay pending — Ollama vision deadlocks on JP7" note with: on jetson, captioning runs **in-process** (transformers 4-bit on the cu132 GPU) because Ollama's CUDA build has no Orin `sm_87` vision kernels (runs CLIP on CPU); text (planner/chat/embeddings) stays on Ollama. Note the `gcc`/`bitsandbytes`/`accelerate` image deps.
  - §8/§8.1: the caption stage uses a **captioner adapter** (Ollama vs in-process), and the coordinator's `INGEST_CAPTION` resource is that adapter — `load()`/`release()` are the adapter's (warm/evict Ollama on mac, load/free the in-process VLM on jetson).
- [ ] **Step 5:** full suite green (`uv run pytest -q`), lint clean.

---

### Task 7: On-board deploy + verify (board-only)

**Files:** none (verification).

- [ ] **Step 1:** Ask the user to `make run-jetson` (or, with permission, do it over SSH). The jetson image now has gcc + bitsandbytes + accelerate + the in-process captioner.
- [ ] **Step 2:** Watch: the worker's `INGEST_CAPTION` pass loads the in-process VLM (first time downloads ~7.5 GB, then ~28 s load), captions run on the GPU (resource bar shows `ingest_caption · <vlm> · ~2.7 GB`, `tegrastats` GR3D busy), and `caption` stage progresses (0/205 → done). Confirm chat + everything else still works.
- [ ] **Step 3:** Confirm **mac** (separately, if available) still captions via Ollama unchanged.
- [ ] **Step 4:** If the resident/throughput differs from the smoke test, tune `max_new_tokens` / model choice and update §3.1 + `footprint_mb` to match reality (doc stays source of truth).

---

## Self-Review

- **Mac/cloud unchanged:** every profile except jetson selects `OllamaCaptioner`, which runs today's exact `caption_messages`+`client.complete` path; the coordinator's INGEST_CAPTION calls `OllamaCaptioner.load()` = `client.warm(model)` — same as plan 13's behavior. ✓ (Tasks 1, 4, 5 assert it.)
- **Jetson-only heavy deps:** gcc/bitsandbytes/accelerate added ONLY in `Dockerfile.jetson`; `VLMCaptioner` is only *constructed* off-board (never `.load()`-ed) so the suite never imports torch-heavy paths. ✓
- **Adapter symmetry:** both adapters implement `caption/load/release/footprint_mb/name/caption_model`; cancellation uses `InferenceCancelled` in both, mapped to `Preempted` by the one stage. ✓
- **Coordinator invariants (plan 13) intact:** lease/piggyback/hard-preempt/robust-release unchanged; only residency resolution generalized to `_resolve(member)`. ✓
- **Packaging:** `captioning` added to both Dockerfiles' COPY + `pyproject only-include` (the `models/` gap lesson). ✓
- **§9 embeddings:** untouched — caption text still embedded via the Ollama client on every profile (jetson keeps Ollama for text). ✓
