# Models — slots, the catalog, backends, and the caption-embedding decision

The exact model slots, the selectable catalog, per-profile backends, download
mechanics, and the settled caption-embedding decision. Design §4/§4.1 carries the
selection *narrative* and *why*; this file is the exact wiring. All of these run
inside the one `models` service (design §5.1), loaded once — never in
`app`/`worker`.

## Slots → default model → backend

| Slot | Default model | Backend (inside the `models` service) | Residency unit |
|---|---|---|---|
| `image_embed` — image and text embeddings, zero-shot tags | SigLIP 2 `so400m-patch14-384` | `TorchWorker` child, `transformers` — MPS (mac) / CUDA (jetson) | `image_embed` |
| `caption` — captions (caption sentence only — no tags) | `gemma4-E2B` GGUF (mac/jetson) · `qwen2.5vl:7b` (cloud) | OpenAI `/v1` call to `llama-server` (mac/jetson) / vLLM (cloud) | `llm_vision` |
| `planner` — query planning, chat answers | `gemma4-E2B` GGUF (mac/jetson) · `qwen2.5:3b` (cloud) | OpenAI `/v1` call to `llama-server` (mac/jetson) / vLLM (cloud) | `llm` |
| `text_embed` — caption text embeddings (§9 similar) | `nomic-embed-text-v1.5` (dedicated text embedder) | `TorchWorker` child, `transformers` — MPS (mac) / CUDA (jetson) | `text_embed` |

Each slot is switchable at runtime from the settings popup (design §4.1, §13);
the table gives the defaults. `caption` and `planner` share one `llama-server`
child, so their two residency units are mutually exclusive.

## The catalog

`models/catalog.py` — plain data, no torch import, the single source of every
selectable model. Fields per entry: `slot`, `key`, `display`, `source`
(HF repo id, or GGUF repo + file + mmproj file), `size_mb` (download),
`cost_mb` + `cost_measured` (resident cost for the governor; `False` means an
estimate, shown as unverified in the UI), `dim` (embedders), `preprocess`
(`input_px`, `resample`, `mode: squash|native`, image embedder only),
`prompt_template` (LLM slots), and `profiles` (which profiles may offer it).

What ships:

| Slot | Entries |
|---|---|
| `image_embed` | `siglip2-so400m-384` (default) · `siglip2-so400m-512` |
| `text_embed` | `nomic-1.5` (default) · `embeddinggemma-300m` · `qwen3-embed-0.6b` |
| `caption` | `gemma4-E2B` (default) · `qwen3-vl-4b` · `qwen3-vl-8b` (mac) |
| `planner` | `gemma4-E2B` (default) · `qwen3-4b-2507` · `qwen3-vl-8b` (mac) |

Everything beyond the defaults carries an **estimated** `cost_mb` until it is
measured on the board the way §8.1 requires (the drop in
`psutil.virtual_memory().available` across a load). Until then the settings popup
labels it unverified, and `test_every_model_fits_its_profile_budget` still refuses
to offer any entry whose `cost_mb + headroom` exceeds the profile budget.

A free-form "type any HF id" field is deliberately **not** offered: `cost_mb`,
`dim` and `preprocess` cannot be discovered at runtime, and each breaks something
different when wrong (OOM / unloadable slot, an unusable vector table, silently
degraded embeddings).

## Switching a slot

`app` owns the choice (`app_settings`, one row per slot) and resolves
**stored → env override → profile default**. On confirm, in one transaction:
write the choice → rebuild `photo_vec` if the new `image_embed` `dim` differs →
`jobs.reprocess()` over the slot's invalidated range — `image_embed`:
`embed`→`taxonomy`, `text_embed`: `caption_embed`, `caption`:
`caption`→`caption_embed`, `planner`: nothing. Then `PUT /models/slots` to the
service, which evicts the outgoing resident and re-registers the slot; the next op
loads the new model.

The `models` service has no DB. It boots on profile defaults and reports a
`generation` with its slots; `app` re-pushes when the resource-bar poll shows a
generation it did not set, so a `models` restart converges without a shared store.

**Preprocessing follows the model.** `GET /embed/spec` returns calibration *and*
the selected image embedder's `preprocess`; `RemoteEmbedder` caches it and resizes
by it (`squash` = stretch to `input_px`², today's behaviour; `native` = fit inside
`input_px`, aspect kept, for NaFlex/native-resolution encoders). No caller holds a
resolution constant.

Since plan 16, captioning and text generation are the **same** `gemma4-E2B` GGUF on
`llama-server` (mac/jetson) — one model, text + vision, on the GPU. Captioning goes
through a **`captioning.Captioner` adapter** (`OpenAICaptioner`) that POSTs an
OpenAI `/v1/chat/completions` request with the image as an `image_url` data-URI,
constrained to `CAPTION_SCHEMA`; the `models` service's `CaptionBackend`
(`modelsvc/backends/caption_backend.py`) wraps it into the caption dict. The
ingest `caption` stage never touches a `Captioner` — it calls the service's
`/caption` over HTTP. There is no in-process caption VLM and no per-profile caption
branch any more (the old `caption_backend` / `caption_model_id` settings are gone).

Caption and planner prompts live in a per-model template registry in
`inference/prompts.py`, keyed by model name, with a shared JSON schema all
templates must satisfy. **Adding a model means adding a template, not touching the
pipeline.** `mac`/`jetson` run `gemma4-E2B` for both caption and planner/chat;
`cloud` stays on Qwen2.5 (vLLM).

## Caption text embedder

The caption text is embedded by a **dedicated text embedder,
`nomic-embed-text-v1.5`** (`config.text_embed_model`), with the model's required
`search_query:` / `search_document:` task prefixes (`embedding/caption_text.py`).
NOT the planner (a chat model has no embedding head) and NOT SigLIP (its text
tower is trained image↔text, so text↔text has no separation — measured). Since
plan 16 dropped Ollama it has no server, so it runs **in-process in the `models`
service** (`embedding/text_embedder.py`, `TextBackend.text_embed`), on MPS (mac) /
CUDA (jetson). The resulting `caption_vec` is a **text-meaning retrieval index**,
consumed as **top-k KNN** (never a fixed cosine floor).

> **DECISION — caption embeddings are the scalable semantic-retrieval index; do not relitigate.**
> `caption_vec` exists to **vector-search meaning-similar captions** (top-k KNN) so a query
> pulls a handful of candidates out of a library of millions **without the LLM ever looping
> over rows**. It is NOT an LLM "read each caption and decide" step, and NOT a rerank
> guardrail — it is the retrieval index. Consequences, settled and measured:
>
> 1. **Embedder = a dedicated text embedder (`nomic-embed-text`), with the required
>    `search_query:` / `search_document:` prefixes.** SigLIP's text tower was tried and
>    **measured inadequate**: it is trained for *image↔text*, so *text↔text* cosines collapse
>    into a ~0.2–0.45 band with no separation (an ideal "red jacket" match scored 0.41 vs an
>    irrelevant "police car" 0.32). nomic ranks the ideal match clearly on top. **Do not go
>    back to SigLIP for caption text.** SigLIP stays for image↔text search and visual similarity.
>    Benchmarked on the real caption corpus, `nomic` / `mxbai-embed-large` / `bge-m3` **tie on
>    retrieval** (recall@5 0.80, recall@10 0.93, MRR ~1.0); `nomic` wins on **size (274 MB)** and
>    **latency (~46 ms/cap)**, decisive for the 8 GB Jetson — so it is the default.
> 2. **Consume as top-k KNN, NOT a fixed cosine floor.** nomic has a high baseline cosine
>    (unrelated captions ≈ 0.4–0.5), so a fixed floor like `RERANK_FLOOR = 0.4` is meaningless
>    here — rank by similarity and take the nearest k; let the agent/LLM verify that shortlist.
> 3. Runs **in-process in the `models` service** (`embedding/text_embedder.py`) on mac and
>    jetson via `transformers`, reached over the models gateway (`/text/embed`) with the same
>    `InferenceClient.embed` call shape as before. It measures **2.14 GB** resident on the Jetson
>    (its `TorchWorker` child, torch + CUDA context included) — NOT the ~0.3 GB the model
>    weights suggest, because it loads with `.to()` and never frees the host copy
>    (design §8.1). It does not ride alongside SigLIP; the governor swaps them. On cloud (vLLM) it keeps the OpenAI
>    `/embeddings` path.
>
> Status: **implemented** — `caption_vec` is written by `backfill_caption_vectors`
> (pipeline group 2c) with `nomic-embed-text-v1.5` and now feeds **§9 "similar photos"**
> (caption-meaning between two photos) and the `/library` search fusion signal.
> **Chat photo-search no longer uses it** — per **plan 17** chat finds photos with SigLIP
> **image↔text** (`search_photos`, design §10), which needs no captions; captions embed the
> whole scene, so caption-text search interleaved scene-neighbours (an SUV beside the real match).

## Why these models

- **Gemma 4 over Gemma 3.** Gemma 3 is dominated on every axis — Gemma 4 12B beats
  Gemma 3 27B by roughly 20 MMMU-Pro points at a third of the memory — and is not used.
- **SigLIP 2 over OpenCLIP** on zero-shot and retrieval. It stays in-process rather
  than behind the inference service because neither `llama-server` nor vLLM exposes
  an image-embedding endpoint. On `mac` it runs host-native on the Apple GPU
  (`embed_device=mps`) — never on the CPU, and never in a container, which gets no
  Metal (design §3.1).

## Model download

On `mac`/`jetson`, the default gemma GGUF (+ mmproj) is fetched once into a
volume/dir — by the `inference` container's entrypoint on jetson, by `make
llama-mac` on mac (design §3.1). SigLIP and the nomic text embedder are fetched
from Hugging Face into a mounted cache the first time they are used. On `cloud`,
vLLM fetches its model from Hugging Face. All these caches survive restarts.

**Any other catalog entry is fetched on demand by the `models` service**
(`modelsvc/downloads.py`): `POST /models/download` starts a background thread —
`huggingface_hub.snapshot_download` for a transformers model, a streaming GET for a
GGUF (+ its mmproj) into the same `/data/models` dir `llama-server` is pointed at.
Progress (`bytes`/`total`, plus a terminal `error`) is kept in a thread-safe dict
and read back through `GET /models/catalog`, which is what the settings popup polls.
A GGUF reports its own bytes as it streams; `snapshot_download` exposes no
callback, so a watcher thread samples the repo's HF cache `blobs/` directory
(partial `.incomplete` files included) and the denominator comes from the repo's
real file sizes — a multi-GB pull stuck at 0% is indistinguishable from a stall.
"Already downloaded" is a probe of the cache path, not a flag — a manually placed
GGUF counts. Downloads never run inside a request, and a download in flight never
blocks inference: it holds no scheduler slot and loads nothing.

## Bake-off gate

A script runs candidate caption models over the same real photos and reports
seconds per photo, memory high-water mark, and side-by-side captions (an open item
is to widen it — dense text, low light, many objects — before the first bulk
ingest). Because published benchmarks are either close or not directly comparable,
the bake-off is how the default is actually chosen. The winner becomes the default
in config, not in code.
