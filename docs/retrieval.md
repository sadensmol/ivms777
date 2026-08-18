# Retrieval — the core interface, planner spec, and scoring detail

The exact `Query` shape, the two-stage signatures, the query-planner output, and
the similar-photo scoring weights/thresholds. Design §9 carries the ADR (hard
filters gate, everything else scores), the two canonical flow diagrams, and the
narrative; this file is the exact interface and numbers. `search/retriever.py` is
the one retrieval pipeline (plan 12); nothing else scores, fuses, narrows, or floors
photos.

## The `Query` and the two stages

```python
Query = {
  text: str | None            # NL query — search, chat, theme discovery
  seed_photo_id: int | None   # a photo — similar, memory "more like this"
  hard_filters: dict          # EXIF facets + explicit date — EXACT, gates (§6.2)
  soft_tags: dict             # {dimension: [label, ...]} — planner hints, SCORE only
  k: int
  weights: dict[str, float] | None   # per-dimension importance, vocab.yaml
  floor: float | None         # caller-set relevance floor; None = rank, don't cut
}
```

Exactly one of `text` / `seed_photo_id` is set. The core exposes its two stages
separately so an interactive caller can paint fast, then refine:

```python
candidates(conn, embedder, owner_id, query) -> [id]                 # phase 1 — FAST
refine(conn, embedder, client, owner_id, query, ids) -> [{id, score, reasons}]  # phase 2 — graceful
retrieve(...) = refine(candidates(...))                              # synchronous convenience
```

The four candidate/contribution mechanisms:

- **Semantic** — query text through the SigLIP text encoder, KNN via `sqlite-vec`,
  inside `candidates()`. Handles "dogs playing in snow" with no matching caption.
- **Keyword** — FTS5 BM25 over captions and tag text, fused into `candidates()`
  alongside semantic. Catches proper nouns, OCR'd text, and exact words embeddings
  smear over.
- **EXIF facets** — exact filters over `photo_facets`, the core's hard pre-filter
  (`_hard_filter`): applied to the candidate list before any scoring.
- **Tag facets** — a sidebar/planner tag hint is a **soft** contribution
  (`_soft_tag_contributions`) scored by `refine()`, never a gate.

`/library` search calls `candidates()` directly for the fused ranking, then narrows
by the sidebar's EXIF/tag filters. `similar`, chat, and memory call the fuller
`refine(candidates())`.

## Query planner output

The planner model (Qwen2.5-3B today, §4) converts a natural-language query into a
`QuerySpec` in one call:

```
"moody shots of the dog at the beach last summer, shot wide open at night"
  -> {"semantic": "dog on a beach",
      "date_from": "2025-06-01", "date_to": "2025-08-31",
      "tags": {"vibe": ["moody"], "setting": ["beach"]},
      "facets": {"time_of_day": ["night"], "aperture": {"lte": 2.0}}}
```

The `facets` block maps directly onto `photo_facets` — categorical keys take a list
of accepted values, numeric keys take `gte`/`lte` bounds. A free-text query is
planned once and its predicates are materialized into the same filter params the
sidebar uses (`f_`/`n_`/`t_`, plus `date_from`/`date_to` over `shot_at`); the chips
are those params, so removing a chip drops a predicate and re-runs the ordinary
filtered search — the planner does not run again until a new query is typed. A wrong
facet guess is visible and removable rather than silently skewing the ranking.

## Similar-photo scoring

`similar_photos` is a thin wrapper over `refine(candidates(Query(seed_photo_id=…)))`
— what follows is the core's seed-query scoring. It degrades gracefully with the
pipeline:

| A photo has… | Similar is computed from… |
|---|---|
| no embedding yet | nothing — it can't be compared |
| embedding only | image-vector KNN (cosine ≥ `similar_min_cosine`, default 0.8) |
| + taxonomy | ⊕ shared **tags across every dimension**, each weighted by that dimension's importance |
| + captions | ⊕ **caption meaning** (caption text-embedding cosine) |

Every matching facet is a scored **contribution**:

1. **Shared tags — all dimensions, per-dimension weighted.** Each shared tag
   contributes `dimension_weight × agreement × idf`, where the **per-dimension
   weight** lives in `vocab.yaml` (`similar_dimension_weights`). `agreement` is the
   weaker of the two confidences and `idf` (0–1) damps common tags. This stops a rare
   `palette=earthy` from outweighing the actual subject.

   The weights order the dimensions **what → where → when → how it looks**, then
   discount each by how reliable SigLIP actually is on it: `subject` 3.0, `setting`
   2.0, `occasion` 1.2, `light` 0.5, `season_weather` 0.4, `palette`/`vibe` 0.3,
   `composition`/`emotion` 0.2, `quality` **0 — ignored**. The low ones are measured,
   not guessed: SigLIP's per-dimension softmax always crowns a winner, so **every
   photo carries every dimension** whether or not it applies — `vibe: chaotic` lands on
   50% of the library, and `emotion` reaches 0.5 confidence on just **15%** of its tags
   (`surprised` on half the photos, objects included). `subject`, by contrast, clears
   0.5 on **79%**. `setting` gets *more* weight than its confidence suggests for the
   opposite reason: 23 well-spread labels but only 22% above 0.5, so without the weight
   a real match never surfaces. Since `idf` already damps a label carried by most
   photos, these weights mostly govern each dimension's **rare** labels — which is
   exactly where an unreliable dimension does its damage.

   These numbers are what they are *after* the §7 prompt work; before it, `subject`
   was right 37% of the time and `composition: top-down` alone landed on 82% of the
   library. Weights cannot rescue a mislabelled tag — fix the labels first.
2. **Caption meaning.** SigLIP tagging is single-label and picks the *dominant*
   subject, so a dog riding in a car is tagged `vehicle` and never shares
   `subject=dog` with a dog on a rooftop. So each caption is embedded with a text
   model (§4) when written, and similarity is the **cosine between caption
   embeddings** — "a dog on a rooftop" ≈ "a dog in a car", while "a small teddy bear"
   ≠ "a small domino tile". Contributes above `similar_caption_min` (default 0.6),
   at `caption_weight (2.0) × strength`.

   **`strength` is the cosine rescaled against its noise floor**, not the raw cosine:
   `(cosine − similar_caption_min) / (1 − similar_caption_min)`, so the bar is 0 and
   an identical caption is 1. Text-embedding cosines do not start at 0 — measured
   over 4k random pairs of the real library, two **unrelated** captions score a
   median **0.62** (p10 0.56, p90 0.72), i.e. 64% of all random pairs clear the 0.6
   bar. Fed in raw, that floor alone earned more than any idf-damped tag could, so
   the caption led the ranking on essentially every photo (it was the top reason on
   **51%** of results, against 18% for all tags combined) — the "captions give bad
   similar photos" symptom. Tags are damped by idf; this is the caption's equivalent.
   After rescaling the same sample gives caption 12%, `subject` 21%, visual 36%, and
   a caption that genuinely means the same (cosine 0.90 → strength 0.75) still leads.
   The **displayed match %** is this strength too, so the panel never shows an
   unrelated pair as an 80% caption match.
3. **Image-vector cosine.** A mild look-alike signal (cosine ≥ `similar_min_cosine`,
   default **0.8**) — how alike two photos *look*, not what they are — and the sole
   signal before taxonomy exists. The floor is high because SigLIP image cosines have
   a high baseline: any two photos sit ~0.5–0.65, so a lower floor admits noise (a
   teddy bear "looks alike" a selfie at 0.63), while genuinely-alike photos are
   0.85–0.98.

**Content gate.** A candidate is "similar" ONLY if it shares a **content** signal that
is also **confident**: a `subject` tag at **≥ `_CONTENT_TAG_MIN` (0.5) agreement**, a
caption at **≥ `_CAPTION_CONTENT_MIN` (0.6) strength**, or a genuine visual near-dup.
SigLIP is single-label over a large vocabulary, so its runner-up subjects sit at
0.2–0.4 and are often simply wrong (blue curtains tagged `subject: screenshot` at
0.39); two photos agreeing on a guess neither of them is share no content. A weak
subject tag still reranks, it just cannot qualify a pair — the same rule as a weak
caption. On the reference library the bar removes ~15% of results and leaves ~9% of
photos with an honest empty strip. Style/scene facets (composition, vibe, palette, light, season,
occasion, setting, emotion, quality) **only rerank** content matches — they never make
two photos similar on their own, and neither does a *weak* caption match: "a dog stands
on a ledge overlooking a cityscape" and "a person stands in front of a dark framed
object" share a 42% caption and nothing else, and used to be called similar.

**Score floor.** A result below `similar_score_min` (default **0.8**) is not shown.
Without it the strip always returned its full `k`, so a photo with nothing genuinely
like it got 12 fillers. The floor is absolute, not relative, and tag contributions
carry `idf` — so in a *tiny* library (a tag on most of the few photos it has) scores
are legitimately small and the strip is legitimately near-empty.

A candidate's score is its contributions **sorted high-to-low and summed with a
decay** (each further facet counts less), so **one strong match — a shared
`subject` — beats a pile of weak ones**. This replaced an earlier flat sum that let
quantity of weak facets win, and a `caption × 3` hack that papered over it. Each
result carries the **reasons** it was chosen, each with a match percentage (the
*weaker* of the two photos' confidences — a 0.71 close-up matching a 1.00 close-up
agree at 71%, never the candidate's raw score). The UI shows the **top 3** reasons —
picked by relevance (contribution = importance × rarity × match), then **displayed
highest match % first**, so the biggest number is always on top — overlaid on each
enlarged thumbnail (§13). The `/photo` "why similar" table sorts the same way, by
match %, with contribution only breaking ties. Pure image-vector KNN as the *primary*
signal was rejected (a dog on a rooftop returned other rooftops); an LLM reranker was
rejected for this interactive path (it reintroduces per-click latency §9.1 forbids).

## Async paint (`/photo`)

**Shipped (plan 12, task 3b):** `/photo/{id}` no longer waits on `similar_photos` to
render — image, EXIF, tags, sources, and the collection collage render on the first
response. The "Similar photos" section ships as an HTMX placeholder
(`hx-trigger="load"`) that fires `GET /photo/{id}/similar`; that route runs the full
`similar_photos` (`refine(candidates())`, with the same collection-member exclusion
as the main route) and returns the strip as a fragment. First paint never waits on
the full-library scan.

**Still future:** splitting that fragment into phase-1 instant KNN order then a
phase-2 reasoned reorder. Today `/photo/{id}/similar` runs the whole
`refine(candidates())` in one call; the `candidates()`/`refine()` split exists to
make the finer two-phase paint possible without a new code path, but that split has
not landed.
