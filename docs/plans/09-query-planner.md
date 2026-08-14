# Query Planner + Parsed-Filter Chips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a free-text library query into a `QuerySpec` (semantic term + date range + facet/tag predicates) with one LLM call (§9.1), apply those predicates alongside the sidebar filters, and show them as removable chips in the top bar (§13). The planner is strictly an enhancement: any failure falls back to raw semantic+keyword fusion.

**Architecture:** The planner is a single structured call whose output is *materialized* into the existing filter param space. `/library?q=…` (unplanned) runs the planner once, then 303-redirects to `/library?<f_/n_/t_/date_from/date_to>&q=<semantic>&planned=1`. From there the page is driven purely by params — chips are just the active predicates, and removing one drops a param and re-runs through the existing filter machinery. No re-planning happens on chip edits, so a wrong guess is correctable and the user's place survives navigation (bfcache).

**Tech Stack:** Pydantic (`QuerySpec`), the existing `InferenceClient` (planner model, no JSON-schema — lenient parse with fallback), FastAPI/Jinja/HTMX, the existing `search/facets.py` + `search/tags.py` filter builders.

**Spec:** `docs/design.md` §9 (retrieval), §9.1 (query planner — the authority for this plan), §6.2 (EXIF facets), §13 (`/library` top bar chips).

## Global Constraints

- **Planner is optional (§9.1).** If it times out, errors, or returns unparseable JSON, fall back to `QuerySpec(semantic=<raw query>)`. Never let a planner failure break search.
- **No JSON-schema on the planner call.** §9.1 explicitly tolerates invalid JSON via the fallback, and Ollama's strict-schema support does not cover dynamic-key maps. Ask for JSON in the prompt, `json.loads` leniently, validate into `QuerySpec`, fall back on any error.
- **Only known keys survive.** Drop any tag dimension not in `vocab.dimensions` and any facet key not in `FACET_KEYS` when materializing — a hallucinated key must not reach SQL.
- **The owner does all git commits.** Never run `git commit`/`git add`. Each task ends at a green suite.
- **Owner-scoped, flat layout, fakes in tests** — as elsewhere in the project.
- Run tests with `uv run pytest`.

---

### Task 1: Design note (§9.1)

Doc-first gate. §9.1 already specifies the planner; add one sentence naming the two concrete mechanics this plan introduces so the doc matches the code.

**Files:**
- Modify: `docs/design.md` (§9.1)

- [ ] **Step 1: Append to the §9.1 paragraph that ends "…visible and correctable."**:

```
Concretely, a free-text query is planned once and its predicates are
materialized into the same filter params the sidebar uses (`f_`/`n_`/`t_`, plus
`date_from`/`date_to` over `shot_at`); the chips are those params, so removing a
chip simply drops a predicate and re-runs the ordinary filtered search — the
planner does not run again until a new query is typed.
```

- [ ] **Step 2: Re-read §9.1** and confirm nothing else now disagrees with the code this plan builds.

---

### Task 2: `QuerySpec` and the planner prompt

**Files:**
- Create: `search/planner.py` (model + prompt only in this task)
- Test: `tests/test_planner_spec.py`

**Interfaces:**
- Produces:
  - `class QuerySpec(BaseModel)` with `semantic: str = ""`, `date_from: str | None = None`, `date_to: str | None = None`, `tags: dict[str, list[str]]`, `facets: dict[str, object]`.
  - `CATEGORICAL_FACETS: tuple[str, ...]`, `NUMERIC_FACETS: tuple[str, ...]` — the facet keys the planner may emit, split by kind (drawn from §6.2).
  - `planner_messages(query: str, dimensions: list[str]) -> list[ChatMessage]`.

- [ ] **Step 1: Write the failing test** `tests/test_planner_spec.py`:

```python
from inference.client import ChatMessage
from search.planner import QuerySpec, planner_messages


def test_query_spec_defaults_are_empty():
    spec = QuerySpec()
    assert spec.semantic == "" and spec.tags == {} and spec.facets == {}
    assert spec.date_from is None and spec.date_to is None


def test_query_spec_parses_the_9_1_example():
    spec = QuerySpec.model_validate({
        "semantic": "dog on a beach",
        "date_from": "2025-06-01", "date_to": "2025-08-31",
        "tags": {"vibe": ["moody"], "setting": ["beach"]},
        "facets": {"time_of_day": ["night"], "aperture": {"lte": 2.0}},
    })
    assert spec.semantic == "dog on a beach"
    assert spec.tags["vibe"] == ["moody"]
    assert spec.facets["aperture"] == {"lte": 2.0}


def test_planner_messages_list_dimensions_and_ask_for_json():
    msgs = planner_messages("moody dog at the beach at night", ["vibe", "setting"])
    system = msgs[0]["content"]
    assert "json" in system.lower()
    assert "vibe" in system and "setting" in system   # allowed tag dimensions
    assert "aperture" in system                        # a numeric facet key
    user: ChatMessage = msgs[1]
    assert "moody dog at the beach at night" in user["content"]
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_planner_spec.py -v`
Expected: FAIL — `search.planner` missing.

- [ ] **Step 3: Implement the model + prompt** in `search/planner.py`:

```python
from pydantic import BaseModel, Field

from inference.client import ChatMessage

# Facet keys the planner may emit, split by kind (§6.2). Categorical map to a
# list of accepted values; numeric map to {gte?, lte?} bounds.
CATEGORICAL_FACETS: tuple[str, ...] = (
    "camera_make", "camera_model", "lens", "software", "flash",
    "exposure_program", "metering_mode", "white_balance",
    "weekday", "time_of_day", "is_weekend",
    "has_gps", "place_city", "place_country", "aspect", "orientation",
)
NUMERIC_FACETS: tuple[str, ...] = (
    "iso", "aperture", "shutter_speed", "focal_length", "exposure_bias",
    "year", "month", "hour", "megapixels",
)


class QuerySpec(BaseModel):
    semantic: str = ""
    date_from: str | None = None
    date_to: str | None = None
    tags: dict[str, list[str]] = Field(default_factory=dict)
    facets: dict[str, object] = Field(default_factory=dict)


def planner_messages(query: str, dimensions: list[str]) -> list[ChatMessage]:
    system = (
        "You convert a photo-search query into a JSON object with keys: "
        '"semantic" (the visual subject to match, a short phrase), '
        '"date_from"/"date_to" (ISO dates or null), '
        '"tags" (an object mapping any of these dimensions to a list of labels: '
        f"{', '.join(dimensions)}), and "
        '"facets" (an object; categorical keys take a list of values, numeric '
        "keys take an object with gte/lte). "
        f"Categorical facet keys: {', '.join(CATEGORICAL_FACETS)}. "
        f"Numeric facet keys: {', '.join(NUMERIC_FACETS)}. "
        "Use only these keys. Omit anything the query does not mention. "
        "Reply with ONLY the JSON object — no prose, no markdown fences."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_planner_spec.py -v`
Expected: PASS.

---

### Task 3: `plan()` with fallback, and `spec_to_params()`

**Files:**
- Modify: `search/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `QuerySpec`, `planner_messages`, `InferenceClient.complete`, `ingest.facets.FACET_KEYS`.
- Produces:
  - `plan(client, model: str, query: str, dimensions: list[str]) -> QuerySpec` — one `complete()` call, lenient JSON parse; on ANY exception or invalid JSON returns `QuerySpec(semantic=query)`.
  - `spec_to_params(spec: QuerySpec, *, query: str, dimensions: list[str]) -> dict[str, str]` — materialize into filter params: `q`, `t_<dim>`, `f_<key>`, `n_<key>=gte:lte`, `date_from`, `date_to`, and always `planned="1"`. Unknown dims/keys dropped.

- [ ] **Step 1: Write the failing test** `tests/test_planner.py`:

```python
import json

from inference.fakes import FakeInferenceClient
from search.planner import plan, spec_to_params


def test_plan_parses_model_json():
    payload = json.dumps({"semantic": "dog on a beach",
                          "facets": {"aperture": {"lte": 2.0}}})
    spec = plan(FakeInferenceClient(responses=[payload]), "m", "moody dog", ["vibe"])
    assert spec.semantic == "dog on a beach"
    assert spec.facets["aperture"] == {"lte": 2.0}


def test_plan_falls_back_on_garbage():
    spec = plan(FakeInferenceClient(responses=["not json at all"]), "m", "sunset", ["vibe"])
    assert spec.semantic == "sunset"        # raw query preserved
    assert spec.facets == {} and spec.tags == {}


def test_plan_falls_back_when_client_raises():
    spec = plan(FakeInferenceClient(responses=[]), "m", "sunset", ["vibe"])  # empty -> asserts
    assert spec.semantic == "sunset"


def test_spec_to_params_materializes_known_predicates():
    from search.planner import QuerySpec
    spec = QuerySpec(
        semantic="dog on a beach", date_from="2025-06-01", date_to="2025-08-31",
        tags={"vibe": ["moody"], "bogus_dim": ["x"]},
        facets={"time_of_day": ["night"], "aperture": {"lte": 2.0}, "bogus_key": ["y"]},
    )
    params = spec_to_params(spec, query="moody dog", dimensions=["vibe", "setting"])
    assert params["q"] == "dog on a beach"
    assert params["planned"] == "1"
    assert params["t_vibe"] == "moody"
    assert "t_bogus_dim" not in params           # unknown dimension dropped
    assert params["f_time_of_day"] == "night"
    assert params["n_aperture"] == ":2.0"        # gte empty, lte 2.0
    assert params["date_from"] == "2025-06-01" and params["date_to"] == "2025-08-31"
    assert "f_bogus_key" not in params and "n_bogus_key" not in params


def test_spec_to_params_uses_raw_query_when_semantic_empty():
    from search.planner import QuerySpec
    params = spec_to_params(QuerySpec(), query="birthday cake", dimensions=["vibe"])
    assert params["q"] == "birthday cake" and params["planned"] == "1"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL — `plan`/`spec_to_params` missing.

- [ ] **Step 3: Implement** — append to `search/planner.py`:

```python
import json

from ingest.facets import FACET_KEYS
from inference.client import InferenceClient


def plan(client: InferenceClient, model: str, query: str, dimensions: list[str]) -> QuerySpec:
    """Free-text query -> QuerySpec in one call. Any failure -> raw-query fallback."""
    try:
        raw = client.complete(model, planner_messages(query, dimensions), timeout=20.0)
        return QuerySpec.model_validate(json.loads(_strip(raw)))
    except Exception:  # noqa: BLE001 - the planner is optional; degrade to fusion (§9.1)
        return QuerySpec(semantic=query)


def _strip(text: str) -> str:
    # Tolerate a stray ```json fence or leading prose around the object.
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _range(bounds: object) -> str | None:
    if not isinstance(bounds, dict):
        return None
    gte, lte = bounds.get("gte"), bounds.get("lte")
    if gte is None and lte is None:
        return None
    return f"{'' if gte is None else gte}:{'' if lte is None else lte}"


def spec_to_params(spec: QuerySpec, *, query: str, dimensions: list[str]) -> dict[str, str]:
    params: dict[str, str] = {"q": spec.semantic.strip() or query, "planned": "1"}
    for dimension, labels in spec.tags.items():
        if dimension in dimensions and labels:
            params[f"t_{dimension}"] = ",".join(labels)
    for key, value in spec.facets.items():
        if key not in FACET_KEYS:
            continue
        if isinstance(value, list) and value:
            params[f"f_{key}"] = ",".join(str(v) for v in value)
        else:
            rng = _range(value)
            if rng is not None:
                params[f"n_{key}"] = rng
    if spec.date_from:
        params["date_from"] = spec.date_from
    if spec.date_to:
        params["date_to"] = spec.date_to
    return params
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_planner.py -v`
Expected: PASS.

---

### Task 4: Date-range filter over `shot_at`

The sidebar has no date range; the planner's `date_from`/`date_to` need plumbing. A tiny pure builder, mirrored on `search/facets.build_where`.

**Files:**
- Create: `search/dates.py`
- Test: `tests/test_date_filter.py`

**Interfaces:**
- Produces: `date_where(params: dict[str, str]) -> tuple[str, list]` — a WHERE fragment to splice after `… FROM photos p WHERE …`, bounding `p.shot_at` by `date_from`/`date_to` (inclusive; `date_to` extended to end-of-day). Empty when neither is present or valid.

- [ ] **Step 1: Write the failing test** `tests/test_date_filter.py`:

```python
from search.dates import date_where
from tests.factories import add_photo


def _ids(conn, where, params):
    return {r["id"] for r in conn.execute(
        "SELECT id FROM photos p WHERE owner_id = 1" + where, params
    )}


def test_date_where_bounds_shot_at(conn):
    a = add_photo(conn, content_hash="a" * 64, shot_at="2025-07-01T10:00:00")
    b = add_photo(conn, content_hash="b" * 64, shot_at="2025-09-01T10:00:00")
    where, params = date_where({"date_from": "2025-06-01", "date_to": "2025-08-31"})
    got = _ids(conn, where, params)
    assert a in got and b not in got


def test_date_where_empty_without_bounds(conn):
    assert date_where({}) == ("", [])
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_date_filter.py -v`
Expected: FAIL — `search.dates` missing.

- [ ] **Step 3: Implement** `search/dates.py`:

```python
def date_where(params: dict[str, str]) -> tuple[str, list]:
    """Bound photos.shot_at by date_from/date_to (inclusive). ISO date strings
    sort lexicographically, so plain string comparison is correct."""
    fragment = ""
    bound: list = []
    date_from = (params.get("date_from") or "").strip()
    date_to = (params.get("date_to") or "").strip()
    if date_from:
        fragment += " AND p.shot_at >= ?"
        bound.append(date_from)
    if date_to:
        fragment += " AND p.shot_at <= ?"
        bound.append(date_to + "T23:59:59")
    return fragment, bound
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_date_filter.py -v`
Expected: PASS.

---

### Task 5: Wire the planner + date filter + chips into `/library`

**Files:**
- Modify: `web/app.py`
- Create: `web/templates/_chips.html`
- Modify: `web/templates/library.html` (render chips)
- Modify: `web/static/app.css` (chip styles)
- Test: `tests/test_web_planner.py`

**Interfaces:**
- Consumes: `search.planner.plan`, `search.planner.spec_to_params`, `search.dates.date_where`, `vocab.dimensions`.
- Behaviour:
  - `GET /library?q=<text>` **without** `planned` → run `plan()`, redirect 303 to `/library?<spec_to_params>`.
  - `GET /library?...&planned=1` (or no `q`) → existing filtered search, now also AND-ing `date_where`, plus chips in the toolbar.
  - `parsed_chips(params) -> list[dict]` builds `{label, remove}` from active `f_`/`n_`/`t_`/`date_from`/`date_to`; `remove` is the current query string minus that one predicate (keeps `planned=1`).

- [ ] **Step 1: Write the failing test** `tests/test_web_planner.py`:

```python
import json

import pytest
from fastapi.testclient import TestClient

from config import Settings
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def planner_client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    spec = json.dumps({"semantic": "beach", "facets": {"time_of_day": ["night"]}})
    monkeypatch.setattr(
        Settings, "build_inference_client",
        lambda self: (FakeInferenceClient(responses=[spec]), "fake"),
    )
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    pid = add_photo(conn, photo_id=1, content_hash="beach", thumb_key="beach.jpg")
    write_vector(conn, pid, fe.embed_texts(["beach"])[0])
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (1, 'time_of_day', 'night')"
    )
    with TestClient(app) as tc:
        yield tc


def test_query_redirects_to_materialized_params(planner_client):
    r = planner_client.get("/library?q=moody+beach+at+night", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers["location"]
    assert "planned=1" in location and "f_time_of_day=night" in location and "q=beach" in location


def test_planned_page_shows_a_removable_chip(planner_client):
    body = planner_client.get("/library?q=beach&f_time_of_day=night&planned=1").text
    assert "time_of_day" in body and "night" in body      # the chip label
    assert 'class="chip' in body                            # rendered as a chip
    assert "/thumb/1" in body                               # the matching photo survives the filter


def test_planner_failure_still_searches(settings, monkeypatch):
    # No monkeypatch of inference -> the default empty fake raises -> fallback.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    add_photo(conn, photo_id=1, content_hash="beach", thumb_key="beach.jpg")
    write_vector(conn, 1, fe.embed_texts(["beach"])[0])
    with TestClient(app) as tc:
        assert "/thumb/1" in tc.get("/library?q=beach").text   # follows redirect, still finds it
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_web_planner.py -v`
Expected: FAIL — no redirect / no chips yet.

- [ ] **Step 3: Add imports** in `web/app.py` (near the other `search` imports):

```python
from search.dates import date_where
from search.planner import plan, spec_to_params
```

- [ ] **Step 4: AND the date filter into `_filter_where`** in `web/app.py` — extend the existing helper so planned date bounds narrow every query:

```python
    def _filter_where(ctx: AppContext, params: dict[str, str]) -> tuple[str, list]:
        # EXIF facets AND model tags AND date range AND the optional dupes flag.
        facet_where, facet_params = build_where(parse_filters(params))
        tags_where, tags_params = tag_where(
            parse_tag_filters(params), ctx.settings.tag_score_min
        )
        date_frag, date_params = date_where(params)
        where = facet_where + tags_where + date_frag
        if params.get("dupes"):
            where += DUPES_ONLY
        return where, [*facet_params, *tags_params, *date_params]
```

- [ ] **Step 5: Add the chips helper and the planner redirect** in `web/app.py`. Put `parsed_chips` next to `_sidebar`, and change the `library` route:

```python
    def parsed_chips(params: dict[str, str]) -> list[dict]:
        from urllib.parse import urlencode

        def without(name: str, value: str | None = None) -> str:
            keep = dict(params)
            if value is not None:  # multi-value f_/t_: drop just this value
                rest = [v for v in keep[name].split(",") if v and v != value]
                if rest:
                    keep[name] = ",".join(rest)
                else:
                    keep.pop(name, None)
            else:
                keep.pop(name, None)
            return "/library?" + urlencode(
                {k: v for k, v in keep.items()
                 if k.startswith(("f_", "n_", "t_")) or k in ("q", "sort", "dupes",
                    "date_from", "date_to", "planned")}
            )

        chips: list[dict] = []
        for name, raw in params.items():
            if name.startswith(("f_", "t_")):
                for value in raw.split(","):
                    if value:
                        chips.append({"label": f"{name[2:]}: {value}",
                                      "remove": without(name, value)})
            elif name.startswith("n_"):
                chips.append({"label": f"{name[2:]}: {raw}", "remove": without(name)})
            elif name in ("date_from", "date_to"):
                chips.append({"label": f"{name}: {raw}", "remove": without(name)})
        return chips
```

Then in the `library` route, add the redirect before rendering:

```python
    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        params = _params(request)
        query = params.get("q", "").strip()
        if query and not params.get("planned"):
            ctx = context()
            client, _ = ctx.settings.build_inference_client()
            spec = plan(client, ctx.settings.planner_model or "fake", query,
                        list(vocab.dimensions))
            target = spec_to_params(spec, query=query, dimensions=list(vocab.dimensions))
            from urllib.parse import urlencode
            return RedirectResponse("/library?" + urlencode(target), status_code=303)
        rows = fetch_page(0, params)
        return templates.TemplateResponse(
            request,
            "library.html",
            {
                "photos": rows,
                "next_offset": len(rows),
                "page_size": context().settings.page_size,
                "query": _query_string(params),
                "sidebar": _sidebar(),
                "tag_sidebar": _tag_sidebar(),
                "chips": parsed_chips(params),
                "active": params,
            },
        )
```

Note the return type annotation is dropped so the route may return `RedirectResponse` or `HTMLResponse`.

- [ ] **Step 6: Keep `planned`/`date_*` in `_query_string`** so pagination and the photo-return round-trip preserve them:

```python
    def _query_string(params: dict[str, str]) -> str:
        keep = {
            k: v for k, v in params.items()
            if k.startswith(("f_", "n_", "t_"))
            or k in ("sort", "q", "dupes", "date_from", "date_to", "planned")
        }
        return urlencode(keep)
```

- [ ] **Step 7: Create** `web/templates/_chips.html`:

```html
{% if chips %}
<div class="chips">
  {% for chip in chips %}
    <span class="chip">{{ chip.label }}<a class="chip-x" href="{{ chip.remove }}">×</a></span>
  {% endfor %}
</div>
{% endif %}
```

- [ ] **Step 8: Render chips** in `web/templates/library.html` — inside the toolbar, after the dupes toggle:

```html
<div class="library-toolbar">
  {% if active.get('dupes') %}
    <a class="toggle on" href="/library">Showing duplicates only — show all</a>
  {% else %}
    <a class="toggle" href="/library?dupes=1">Show duplicates only</a>
  {% endif %}
  {% include "_chips.html" %}
</div>
```

- [ ] **Step 9: Add chip styles** — append to `web/static/app.css`:

```css
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.chip { background: #1e1e1e; border: 1px solid #333; border-radius: 12px;
  padding: 2px 8px; font-size: 12px; color: #ddd; }
.chip-x { margin-left: 6px; color: #888; text-decoration: none; }
.chip-x:hover { color: #fff; }
```

- [ ] **Step 10: Run the planner route tests, verify pass**

Run: `uv run pytest tests/test_web_planner.py -v`
Expected: PASS.

- [ ] **Step 11: Run the full suite**

Run: `uv run pytest`
Expected: PASS — existing `tests/test_web_search.py` still green (its `?q=` requests now 303 through the fallback planner and TestClient follows the redirect to the same results).

---

## Self-Review

**Spec coverage (§9.1):**
- QuerySpec from one call — Task 2 (`QuerySpec`, `planner_messages`), Task 3 (`plan`). ✓
- `facets` block maps to `photo_facets` (categorical list / numeric gte-lte) — Task 3 `spec_to_params` → `f_`/`n_`, applied by existing `build_where`. ✓
- Exact, removable chips — Task 5 `parsed_chips` + `_chips.html`; removal drops a param, no re-plan. ✓
- Strictly an enhancement; invalid JSON/timeout/failure → raw fusion — Task 3 fallback + `test_planner_failure_still_searches`. ✓
- Interpretation always visible/correctable — chips render the active predicates (Task 5). ✓
- Date range (§9.1 example `date_from`/`date_to`) — Task 4 `date_where`, wired in Task 5 step 4. ✓
- §13 top bar chips — Task 5 steps 7–9. ✓

**Non-goals (kept out of this plan, deliberately):** caption vocabulary mining / tag suggestions (the other half of phase 4) and routing chat through the planner (§10) — separate follow-ups.

**Placeholder scan:** every code step is concrete. ✓

**Type consistency:** `plan(client, model, query, dimensions)` and `spec_to_params(spec, *, query, dimensions)` used identically in Tasks 3 and 5. `date_where(params) -> (str, list)` matches `build_where`/`tag_where` shape and splices the same way in `_filter_where`. Chip `remove` hrefs keep `planned=1`, so no route re-plans on chip edits. ✓
