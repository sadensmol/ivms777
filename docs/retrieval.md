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
| EXIF only, not embedded yet | **moment** — same time, same place (needs no model at all) |
| + embedding | ⊕ image-vector cosine above its gate |
| + taxonomy | ⊕ shared **tags**, by tier |
| + captions | ⊕ **caption meaning** (caption text-embedding cosine) |

### The signals

Every signal is described by exactly two numbers — a **gate** (below it the signal is
*absent*, never a small number) and a **weight** (the most it may claim on its own, on
one shared 0–1 scale). All of them live in `search/signals.py`; the three gates a
deployment may retune are in `config.py`, and the four tier weights are in `vocab.yaml`.

| # | Signal | Gate | Weight | Content? | Why this gate |
|---|---|---|---|---|---|
| 1 | **image** — SigLIP image↔image cosine | **0.80** | **0.95** | ✔ | top 4 % of all pairs; a random pair scores 0.558 |
| 2 | **subject** — shared `subject` tag | **0.80** | **0.75** | ✔ | median top-`subject` is 0.83; 93 % clear 0.5, so 0.5 gates nothing |
| 3 | **caption** — caption-embedding cosine | **0.75** | **0.65** | ✔ | top 5 %; a random pair scores 0.621 |
| 4 | **where** — `setting`, `occasion` | 0.50 | **0.40** | ✘ rerank | 38 % / 51 % of photos clear 0.5 |
| 5 | **moment** — time × place | **0.20** | **0.35** | ✔ | ≈ top 4 % of pairs (below) |
| 6 | **look** — `light`, `season_weather`, `palette`, `vibe`, `composition`, `emotion` | 0.50 | **0.12** | ✘ rerank | least reliable: `emotion` clears 0.5 on 28 % |
| 7 | **quality** — `sharp`, `blurry`, … | 0.50 | **0.03** | ✘ rerank | pure tiebreak; `sharp` is on 201/206 photos |

A text query has no seed image vector, so its **fused rank** stands in for signal 1 at
weight **0.60** — deliberately lighter, because a fused rank says "this matched the
query somehow", not "these two photos are alike".

### The formula

```
cosine-like (image, caption, moment):
    raw < gate   →  absent
    raw ≥ gate   →  evidence = w × (0.5 + 0.5 × (raw − gate) / (1 − gate))

tags:
    agreement < gate  →  absent
    agreement ≥ gate  →  evidence = w × agreement × (0.4 + 0.6 × idf)

score = 1 − Π (1 − evidence)          # noisy-OR, a true 0–1
qualify if ≥1 CONTENT signal present AND score ≥ similar_score_min (0.25)
```

**Why cosines get a 0.5 entry ramp and tags do not.** A cosine's gate must sit well
above the noise, so a pair *just above* it would score ~0 if the strength were rescaled
from the gate — admitting it would buy nothing, and a genuine near-dup at 0.85 would
lose to a tag. The ramp makes crossing a gate immediately worth half the weight. Tag
confidences are already softmax probabilities spanning 0–1, and `idf` is their
equivalent noise floor; the `0.4 + 0.6 × idf` damp keeps a common-but-confident label
at 40 % of its weight instead of zeroing it.

**The rule for setting any gate: never below the library's random-pair median.** This is
the bug the model replaced. `similar_caption_min` was 0.60 — *under* the 0.621 median —
so 64 % of all pairs cleared it and the caption signal scored pure chance. Combined with
a `subject` bar of 0.5 that gated nothing, a girl by a Christmas tree (mis-tagged
`subject: toy` at 0.58, runner-up `person` at 0.31) was scored as similar to a teddy
bear, with `light: low light 69 %` as the headline reason. All four of its signals now
fall below their gates and it is dropped.

**Why tiers instead of ten per-dimension weights.** Ten hand-tuned numbers could not be
reasoned about, and `idf` already handles the part that genuinely varies — how rare a
specific label is. A shared `museum` outranks a shared `indoor` without either needing
a weight of its own. Four tiers, four numbers, in `vocab.yaml`.

These numbers are what they are *after* the §7 prompt work; before it, `subject` was
right 37 % of the time. **No weight can rescue a mislabelled tag** — a carnival ride
confidently tagged `subject: toy` at 0.98 will still match a teddy bear. Fix labels in
`vocab.yaml`, not weights.

### `moment` — same time, same place

`exp(−Δt / 12h) × exp(−Δd / 1km)`, from EXIF `shot_at` and GPS. Same hour + same block
→ 0.75; same afternoon + 500 m → 0.47; six hours + 1 km → 0.22, just admitted; next day
or 5 km → absent. If either photo has no GPS the place is **unknown, not zero**, so it
falls back to time alone × 0.5.

It earns a content-signal weight because closeness in *both* time and place is rare:
measured over the 184 reference photos carrying time and GPS (25 distinct ~1 km places),
the median pair is **886 hours and 119 km apart**, and only **3.75 %** of pairs fall
within 1 h / 200 m — about as selective as an image cosine of 0.80. `similar_score_min`
is sized so a lone `moment` at same-hour/same-block (0.30) survives on its own, while
same-afternoon/500 m (0.23) needs a second signal to agree.

### Content gate, floor, and reasons

**Content gate.** A candidate is "similar" ONLY if it shares a content signal — image,
`subject`, caption, or moment. Clearing the gate *is* the content bar: there is no
second threshold. `where`, `look` and `quality` **only rerank** — they never qualify a
pair, however many of them agree.

**Score floor.** A result below `similar_score_min` (default **0.25** on the 0–1 score)
is not shown. Without it the strip always returned its full `k`, so a photo with nothing
genuinely like it got 12 fillers. On the reference library this leaves 2 % of photos
with an honest empty strip and a median of 9 results.

**Reasons.** Each result carries its **top 3 contributions by evidence, in that order** —
what actually drove the match leads. The match % shown beside each is the *agreement*
(for a tag, the weaker of the two photos' confidences; for a cosine, its strength above
the gate), which answers a different question and is deliberately not the sort key.
Ordering reasons by percentage instead put `quality: sharp 100 %` at the top of **403**
result cards — it agrees perfectly on almost every pair and drives ~0.01. A big number
is not a big reason. The `/photo` "why similar" table sorts the same way, and shows **no
row at all** for a facet below its gate, since it contributed nothing.

Pure image-vector KNN as the *primary* signal was rejected (a dog on a rooftop returned
other rooftops); an LLM reranker was rejected for this interactive path (it reintroduces
per-click latency §9.1 forbids).

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
