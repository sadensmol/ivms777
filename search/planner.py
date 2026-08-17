import json

from pydantic import BaseModel, Field

from inference.client import ChatMessage, InferenceClient

# Facet keys the planner may emit, split by kind (§6.2). Categorical map to a
# list of accepted values; numeric map to {gte?, lte?} bounds. These become HARD
# EXIF gates (§6.2 ADR), so the list is deliberately curated: `aspect` and
# `orientation` are OMITTED — a weak planner hallucinates them for any query
# ("man in black" → orientation:any / aspect:4:3), and because a hard facet is an
# exact cut, that empties the whole result pool before the caption rank runs. They
# stay real *sidebar* facets (a user can still pick them), just never planner-emitted.
CATEGORICAL_FACETS: tuple[str, ...] = (
    "camera_make", "camera_model", "lens", "software", "flash",
    "exposure_program", "metering_mode", "white_balance",
    "weekday", "time_of_day", "is_weekend",
    "has_gps", "place_city", "place_country",
)
NUMERIC_FACETS: tuple[str, ...] = (
    "iso", "aperture", "shutter_speed", "focal_length", "exposure_bias",
    "year", "month", "hour", "megapixels",
)
# Only these keys survive from a planned spec into hard filters — a curated allowlist,
# so a hallucinated or non-planner facet (e.g. `aspect`) never reaches SQL as a gate.
PLANNER_FACET_KEYS: frozenset[str] = frozenset(CATEGORICAL_FACETS) | frozenset(NUMERIC_FACETS)


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
        "Facets are HARD filters — a photo that does not match is excluded — so emit a "
        "facet ONLY when the query names that exact camera/place/time/date attribute. "
        "For a plain visual query like 'man in black' or 'dogs on a beach', facets MUST "
        "be an empty object {} — put the visual subject in \"semantic\" (and colours/"
        "moods/subjects in \"tags\"), NEVER in facets. When unsure, omit. "
        "Reply with ONLY the JSON object — no prose, no markdown fences."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def plan(client: InferenceClient, model: str, query: str, dimensions: list[str]) -> QuerySpec:
    """Free-text query -> QuerySpec in one call. Any failure -> raw-query fallback."""
    try:
        raw = client.complete(model, planner_messages(query, dimensions), timeout=20.0, temperature=0)
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
    """Materialize a QuerySpec into the sidebar's filter param space (§9.1).

    Unknown tag dimensions and facet keys are dropped so a hallucinated key never
    reaches SQL. `planned=1` marks the params as already planned, so the library
    route does not run the planner again on chip edits.
    """
    params: dict[str, str] = {"q": spec.semantic.strip() or query, "planned": "1"}
    for dimension, labels in spec.tags.items():
        if dimension in dimensions and labels:
            params[f"t_{dimension}"] = ",".join(labels)
    for key, value in spec.facets.items():
        if key not in PLANNER_FACET_KEYS:
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
