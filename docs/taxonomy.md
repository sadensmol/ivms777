# Taxonomy — dimensions, scoring, and vocabulary mining

The ten tag dimensions, how each is scored, the tuning knobs, and the
vocabulary-mining job. Design §7 carries the summary and the key decisions; this
file is the exact vocabulary and mechanics. Everything here is editable in
`vocab.yaml` without touching code — re-scoring a changed dimension only re-runs
the SigLIP stage, which is cheap.

## The ten dimensions

| Dimension | Example labels |
|---|---|
| `subject` | person, people, crowd, pet, food, building, nature, vehicle, document, artwork, toy, clothing, product |
| `setting` | indoor, outdoor, beach, mountain, forest, city street, restaurant, home, office, water, snow |
| `vibe` | cozy, energetic, serene, moody, festive, nostalgic, dramatic, minimal, chaotic, romantic |
| `emotion` | joyful, sad, tense, affectionate, playful, contemplative, neutral |
| `light` | golden hour, blue hour, night, harsh midday, overcast, backlit, neon, candlelit |
| `season_weather` | summer, autumn, winter, spring, rain, snow, fog, clear sky |
| `composition` | close-up, selfie, wide shot, aerial, shallow depth of field, symmetry, silhouette, leading lines |
| `palette` | warm, cool, pastel, vivid, monochrome, dark, bright, high contrast |
| `occasion` | birthday, wedding, travel, hike, concert, holiday, everyday, work |
| `quality` | sharp, blurry, noisy, overexposed, underexposed |

## Sources per dimension

- `palette` and `quality` come from cheap pixel statistics first (mean saturation,
  Laplacian variance, histogram clipping). SigLIP refines them.
- All ten get SigLIP zero-shot scores against a **sentence per label**. A label is
  NOT fed to the text tower as its bare name in a template: `vocab.yaml`'s `prompts:`
  block (keyed dimension → label) gives it one or more natural sentences, each
  embedded and then **averaged into one label vector** (prompt ensembling), so a
  single unlucky phrasing cannot decide a tag. A label with no entry falls back to
  its dimension's template in `ingest/taxonomy.py` (`"a photo with a {label} mood"`,
  `"a photo taken in a {label}"`, …).

  **Why this matters more than any weight.** The templates alone produce strings
  SigLIP was never trained on — "a photo of portrait", "a photo of top-down", "a
  photo of work". Scored against a hand-labelled 20-photo sample, template-only
  `subject` was correct **37%** of the time (top-3 68%); the same model with per-label
  sentences reaches **80%** (top-3 100%), and library-wide median confidence went 0.39
  → 0.76. Editing a prompt re-embeds only the label side, so a full re-tag of a
  library is seconds — no image is re-encoded.

  **`subject` labels must be mutually exclusive**, because the softmax picks exactly
  one and §9's content gate rests on it. `portrait`/`selfie`/`candid`/`group of
  people` all meant "people" and split that mass, so eleven photos of one family trip
  landed on four different labels and shared nothing; they collapsed to
  `person`/`people`/`crowd` (how many) with framing moved to `composition: selfie`.
  `clothing` and `product` were added because their absence forced wrong labels — torn
  jeans came out `selfie`, a bottle of motor oil `screenshot`.

  SigLIP is a **per-dimension classifier**, not
  an absolute detector: its raw sigmoid probabilities are tiny (~1e-4) and the same
  across every label, so an absolute floor tags nothing. Instead each dimension is
  scored by a **softmax over its own labels**, and the winning label (plus any
  runner-up within `select_ratio` of it, capped at `max_per_dim`) is kept with that
  softmax probability as its score. Every dimension therefore contributes its best
  guess, and the stored score is a real 0..1 confidence comparable within the
  dimension.
- The caption model (§4) writes **only the caption sentence**, never tags. It used
  to also return its own vocabulary picks (`source='vlm'`), but a free-text VLM guess
  stored at a flat `1.0` confidence overrode SigLIP's real per-dimension scores and
  mislabeled photos (e.g. a plush toy tagged `subject=dog` at 1.0). Tags now come
  **only** from SigLIP + pixel stats + EXIF; the vocabulary already covers concrete
  subjects (`dog`, `cat`, `bird`, …), scored honestly by SigLIP.
- `shot_at`, `camera`, and GPS come from EXIF with `source='exif'`.

## Tuning knobs

`max_per_dim` and `select_ratio` live in `vocab.yaml`. They start at permissive
defaults (top label always, runners-up within half the top's probability, three
labels max) and are tuned against a small hand-labeled dev set of ~100 photos built
during phase 2.

## Vocabulary mining

The starting vocabulary cannot anticipate one particular library. A batch job reads
every caption, extracts recurring noun phrases, and drops any that an existing label
already covers (cosine similarity between SigLIP text embeddings above a threshold).
What remains is ranked by frequency and offered in the UI as "suggested tags", each
with the photos that triggered it.

Accepting a suggestion appends the label to `vocab.yaml` and queues a re-run of the
taxonomy stage for that dimension only. That stage is SigLIP-only, so a new label
costs seconds across the whole library, not hours. This is how the tag vocabulary
grows into the collection instead of being guessed up front.
