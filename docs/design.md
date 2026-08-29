# Photo Library Organizer — Design

Status: approved design, ready for implementation planning
Date: 2026-08-13

## 1. Goal

A Python service with a simple web UI, deployed on a GPU box you control. It
works in two stages.

**Stage 1 — understand, in the cloud.** You select photos in the browser and
upload them. The service classifies every photo, writes a description and tags,
and lets you search, filter, find similar photos, browse suggested groups, and
ask questions about the collection in plain language.

**Stage 2 — reorganize, on your machine.** When the collection is fully
processed, Organize turns the principle on screen into a downloadable **plan
document**, and `ivms777-sync` — a small standalone CLI that runs on the machine
holding the photos — applies it: it creates one folder per album inside your own
folder, moves each file into the folder its album chose, writes each album's
description alongside, and sweeps redundant copies aside. Filenames never change.
It shows you every operation and asks before it touches anything.

The two stages share nothing but that plan document. The cloud never reaches into
your filesystem; the CLI never needs your library uploaded a second time.

All inference runs on hardware you control. Photos never go to a third-party
API.

## 2. Non-goals (v1)

- Face detection and person clustering. Deferred to v2.
- Authentication, signup, and per-user quotas. Deferred to v02; see section 3.2.
- Editing or rating photos in the cloud UI. The only thing that ever writes to
  your disk is `ivms777-sync`, and only on explicit confirmation.
- Cloud model APIs of any kind.
- Video files.

## 3. Constraints and targets

### 3.1 Deploy profiles

The same code runs in three places. `jetson` and `cloud` are containerised — a
base `compose.yaml` plus one per-profile overlay (`compose.jetson.yaml` /
`compose.cloud.yaml`). **`mac` is not containerised at all**: `make up` runs the
models service, the worker and the app as plain host processes, because only a
host process reaches the Apple GPU (see "Why `mac` is host-native" below). The
differences are which inference service is active, which model name is in config,
and — on `jetson` only — which app image is built (see "Jetson image" below).
Ingest is identical everywhere — photos always arrive by upload (section 3.2b),
so no profile depends on a host bind mount.

| Profile | Inference | Caption model | Planner / chat model | Embed device |
|---|---|---|---|---|
| `mac` | `llama-server` on the **host** (Metal) | `gemma4-E2B` | `gemma4-E2B` | `mps` |
| `jetson` | `llama-server` in a container (sm_87 CUDA) | `gemma4-E2B` | `gemma4-E2B` | `cuda` |
| `cloud` | vLLM in a container, `--gpus all` | `qwen2.5vl:7b` | `qwen2.5:3b` | `cuda` |

The model columns are the profile **defaults**, not a fixed roster: since plan 21
each of the four model **slots** (§4.1) is switchable at runtime from the settings
popup (§13), and the stored choice wins over these defaults. `cloud` is the
exception — its slots are config-only.

Since plan 16, `mac` and `jetson` run **one gemma4-E2B GGUF on llama.cpp
`llama-server`**, which serves **both** the caption (vision) and planner/chat
(text) roles over the OpenAI `/v1` API — so both model-name columns are the same
`gemma4-E2B` (the name is for storage/display; `llama-server` serves whatever
`-m` GGUF it loaded). Override with `IVMS777_CAPTION_MODEL` /
`IVMS777_PLANNER_MODEL`. There is **no Ollama** on either profile, and **no
in-process caption VLM** — captioning is a plain OpenAI call to `llama-server`
with the image as an `image_url` data-URI. `cloud` is unchanged by plan 16 (still
vLLM; an open item).

**Why `mac` is host-native — and why NOTHING on it may run on the CPU.** Docker
Desktop on macOS boots a Linux VM, and Apple exposes no GPU to Linux guests —
there is no Metal and no MPS in a container, and no configuration changes that.
Apple's own `container` project does not support GPU passthrough either. A
containerised model on a Mac silently falls back to the CPU: 3-6x slower for
`llama-server`, and **54x** slower for SigLIP (§8.1). That fallback is forbidden
here, so there is **no containerised `mac` path at all** — no `compose.mac.yaml`.
Everything model-related is a host process on the Apple GPU:

- `llama-server`, built natively with Metal (`-DGGML_METAL=ON`), on `:8080` —
  started by `make llama-mac`, reused (not supervised) by the models service.
- The `models` service itself, host-native, with **`embed_device=mps`**: SigLIP
  and the nomic caption-text embedder load through torch's Metal backend.
- `app` and `worker`, host processes reaching the models service at
  `localhost:9000` — they hold no model either way (§5.1).

`make up` starts all of it; the mac profile's defaults (`config.PROFILE_DEFAULTS`)
are the host-native URLs, not container hostnames.

On Linux this problem does not exist. Under `jetson` and `cloud` everything,
including inference, runs in containers with real GPU access via the NVIDIA
container runtime.

**Why llama.cpp and not vLLM on Mac and Jetson.** vLLM's Metal backend does not
support vision models, which is a core workload here, and Docker's own benchmarks
put it 1.2-1.3x behind llama.cpp on Apple Silicon with far more variance. On
Jetson, vLLM's aarch64 build is a maintenance burden — and crucially only a
source-built `sm_87` llama.cpp runs the gemma vision projector on the GPU (§4).
vLLM earns its place only under `cloud`, where continuous batching genuinely helps
with concurrent users.

**Jetson sizing.** After the real, non-reclaimable baseline (kernel + firmware
carveout + system daemons **plus the `app`/`worker`/`models` Python containers**,
~3 GB in practice — higher than the ~1.5 GB kernel-only figure), only **~4.3 GB**
is free for models. gemma4-E2B on the `llama-server` child measures **~5.0 GB**
resident — Q4_K_M weights (~3.1 GB) + the F16 mmproj (~1 GB) + KV/compute buffers +
its own CUDA context — far larger than the SigLIP (~1.6 GB) + nomic (~0.3 GB) torch
models the `models` service hosts in its `TorchWorker` children (§8.1). So **gemma and SigLIP CANNOT be
resident together**: the memory governor (§8.1) evicts SigLIP before it loads gemma
and reloads it afterwards — they **swap** across the embed↔caption stage boundary.
The catalog's per-model cost figures (§4.1, overridable with
`IVMS777_MODEL_COST_MB`) must track these real figures: an under-estimate for gemma makes it skip the eviction and load gemma
on top of SigLIP → OOM.

Everything model-related runs **on the GPU — never the CPU. There is no CPU
offload, on any profile.** gemma must therefore fit **entirely** in GPU RAM:
`llama-server` is launched with **`-ngl 99` (every layer on the GPU)**, and the
conveyor makes room by evicting SigLIP + nomic first (§8.1).

**gemma has two spawn modes, and the vision projector is loaded ONLY for
captioning** — it is the model's vision half, useless to a text turn:

| mode | residency unit | slot | spawn | used by |
|---|---|---|---|---|
| text | `llm` | `planner` | `-m <planner gguf> -ngl 99` | chat, planner |
| vision | `llm_vision` | `caption` | `-m <caption gguf>` **+ `--mmproj`** | captioning |

The two are **mutually exclusive** — one `llama-server` child, one port — so loading
either frees the other. Chat therefore never pays the projector's ~531 MB, which on
the 8 GB board is the difference between ~170 MB and ~700 MB of headroom.

The projector is the **Q8_0** build — `mmproj-gemma-4-E2B-it-Q8_0.gguf` (531 MB) from
`ggml-org/gemma-4-E2B-it-GGUF`, the default since the F16 projector (985 MB) is what
OOM-aborted `llama-server`: its CLIP tensor buffer (589 MiB) was the allocation that
failed. The weights still come from `unsloth/…`, so the projector has its own repo
var (`LLAMA_MMPROJ_REPO`). A small KV context (`IVMS777_LLM_CTX`, jetson 2048) keeps
the rest down. **mac is not memory-bound and does not pay that price: it serves
`-c 32768`** (`make up` passes `LLAMA_CTX` and exports the same number to `app`/
`worker`). Measured on a 32 GB machine, llama-server is 4.11 GiB resident at 4096 and
4.28 GiB at 32768 — gemma's sliding window keeps the KV cache small, so the whole
increase costs ~0.17 GiB. 4096 was not a saving, it was a ceiling: a 40-shot memory's
story prompt is ~6 300 tokens, llama-server **rejects** (never truncates) an
over-long request, and the film failed for want of narration. `make llama-mac` reads
the running server's `/props` and RESTARTS it when its context is not the wanted one —
a `-c` change is the one thing that does not hot-reload, and a stale server looks
exactly like a broken feature. If a pinned all-GPU load still does not fit, `llama-server` aborts at
model load (`failed to fit params to free device memory … abort`) — that is a **hard,
surfaced error** (`app` streams an error turn to the chat UI), and the only fix is
to **shrink the GPU footprint** (smaller KV context, Q8 projector, smaller model
weights), **never** to spill layers to the CPU. mac keeps `-ngl 99` (Metal, ample
RAM). At MAXN_SUPER expect ~5–6 s/photo for captioning with gemma fully on the GPU.

**The empirical reference — device facts, the MAXN_SUPER power-mode lever, the
single-stream decode benchmark, and the GPU-vision benchmark that proved a
source-built `sm_87` `llama.cpp` runs the gemma projector on the GPU (0.57 s/img)
— lives in [`docs/jetson-tuning.md`](jetson-tuning.md).** That measurement is the
basis of plan 16: one gemma4-E2B GGUF on `llama-server` doing text (~30 tok/s) and
vision on the GPU, replacing qwen-text + the in-process VLM, and dropping Ollama
on jetson.

The shipping default on both `mac` and `jetson` is the single **`gemma4-E2B`
GGUF** served by `llama-server` for both roles. It is not an Ollama tag or an HF
`transformers` load any more — `llama-server` loads the GGUF (`-m` + `--mmproj`)
that the entrypoint fetches into the mounted volume on first run (jetson) or
`make llama-mac` downloads to `$HOME/.llama/models` (mac) — a persistent cache
outside the library dir, so `make clean` keeps both the GGUF and the built
`llama.cpp`; only `make llama-rebuild` wipes them. `IVMS777_CAPTION_MODEL`
/ `IVMS777_PLANNER_MODEL` override the name used for storage/display; the actual
weights are whichever GGUF `llama-server` was pointed at. Published benchmark
scores do not settle the captioner choice on their own — the phase 1 bake-off
decides it on real photos.

**Jetson image.** Only the `models` service needs GPU/torch on Jetson —
`app`/`worker` build from the same generic `Dockerfile` as every other profile
(`python:3.12-slim`, CPU-only PyPI `torch`), and never import it: they are thin
HTTP clients (§5.1). The `models` service instead builds from a dedicated
**`Dockerfile.models.jetson`** (repurposed from the old `Dockerfile.jetson`,
which used to build `app`/`worker` directly, back when SigLIP ran in-process
there) — which, on **JetPack 7** (L4T r39, CUDA 13.2), is the *same*
`python:3.12-slim` image as the mac/cloud `Dockerfile.models` with **one
change: `torch` + `torchvision` come from the CUDA-13.2 index**
(`https://download.pytorch.org/whl/cu132`) instead of the CPU PyPI wheel.
JetPack 7 exposes the Orin as **SBSA**, so those upstream CUDA-13.2 wheels run
on the iGPU directly, and the NVIDIA container runtime (`runtime: nvidia`)
hands that iGPU to the two containers that need it — `inference` (the sm_87
`llama-server`, built from `Dockerfile.llamacpp.jetson`) and `models`;
`app`/`worker` declare no GPU access at all. The `models` container
must also declare `NVIDIA_VISIBLE_DEVICES=all` + `NVIDIA_DRIVER_CAPABILITIES=all`:
unlike the CUDA base images, the `python:3.12-slim` base does not set them, and
without them the runtime injects no driver libs, so `libcuda` is absent and
`torch.cuda.is_available()` is `False`. There is **no jetson-containers, no
dusty-nv base image, and no `autotag`**: JetPack 7 ships system Python 3.12,
matching `pyproject.toml`'s `>=3.12` floor, so the image runs the normal `uv
sync --extra models` from `pyproject.toml` and then reinstalls
`torch`/`torchvision` from the cu132 index over the CPU torch that `uv sync`
brought. `numpy` stays on `pyproject`'s `>=2.5.2` — the cu132 wheels are built
against NumPy 2.x, so no Jetson-specific numpy pin is needed.

**Captioning + text both run on the `sm_87` `llama-server`; nothing captions
in-process.** Both roles are one source-built `sm_87` `llama.cpp` (the
GPU-vision rationale is in [`docs/jetson-tuning.md`](jetson-tuning.md)). That
binary is **built once** and reused: on the target board the `inference` compose
service runs the prebuilt `llama-server` from the `llamacpp` volume under the
cu130 image (no in-image recompile), and `Dockerfile.llamacpp.jetson` is the
reproducible from-scratch builder for a fresh board. Captioning is therefore a
plain OpenAI `/v1/chat/completions` call to the `inference` container with the
image as an `image_url` — no `transformers` VLM, no `bitsandbytes`/`accelerate`,
no C toolchain in `Dockerfile.models.jetson`. The `models` service on jetson
loads only SigLIP and the small nomic caption-text embedder on the cu132 GPU.

### 3.2 Multi-tenancy

v1 ships as a single-user product with multi-user bones. There is **no auth, no
login, no user table, no admin role, no signup, and no quotas**. There is one
implicit owner, and `owner_id` is a constant. Deploy it behind a private URL or
a reverse proxy that requires a password.

But three things are cheap now and expensive to retrofit, so they are in from
the start:

- `owner_id` on every user-scoped row and in every query.
- Photo bytes reached through a `Storage` interface (local filesystem now,
  object storage later).
- Inference reached through one HTTP client, so swapping the inference server
  (llama-server / vLLM) is a config change.

v02 adds accounts, signup, and quotas on top of this. Because upload is already
the only ingest path and every row is already owner-scoped, that is an additive
change: no data migration, no rewrite of every query.

### 3.2b Getting photos in — upload

Photos arrive by browser upload. There is no host bind mount, no server-side
folder picker, and no path the server walks. The server never sees your
filesystem; it sees the bytes you chose to send.

The upload screen accepts a whole directory (`<input webkitdirectory>`) or a
drag-and-drop selection, and runs in three steps:

1. **Hash locally.** A Web Worker computes each file's SHA-256, one file at a
   time, off the main thread. WebCrypto has no incremental digest, so a file is
   read whole — handling them one at a time bounds memory to the largest single
   photo rather than the size of the selection. Nothing is sent yet.
   `crypto.subtle` exists only in a *secure context* (HTTPS or `localhost`), but
   the app is normally reached over plain HTTP on the LAN (`http://<jetson>:8000`)
   where it is `undefined`; so the worker falls back to a pure-JS SHA-256 there.
   The fallback digest is byte-identical to the server's `hashlib.sha256`, which
   step 2's probe and step 3's server-side verification both depend on.
2. **Ask what is new.** The client POSTs each hash *with the path it came from*
   to `/api/upload/probe`, which answers with the subset whose bytes the server
   has never seen. Paths for content it already holds are recorded on the spot —
   that is what keeps duplicate detection correct when no bytes move.
   Re-uploading a folder you already sent transfers nothing.
3. **Send the new bytes.** Files upload in bounded concurrent batches with a
   per-file progress row. The relative path from the selected root travels
   alongside each file and is recorded, because that path is what stage 2 needs
   to find the file again on disk.

Interruptions are cheap: closing the tab loses only in-flight files, and
restarting the upload re-probes and resumes with what is still missing. The
server verifies the hash of every received file before storing it, so a
truncated transfer is rejected rather than indexed.

Originals are kept. They are written through the `Storage` interface under a
content-addressed key and are needed for full-resolution viewing and for
re-processing the library when a better model arrives.

**Why upload rather than a host mount.** A mounted folder is faster and copies
nothing, but it only works when the app runs on the same machine as the photos.
That constraint defeats the point of a hosted GPU. Upload costs disk and a first
transfer; in exchange, ingest is one code path on every profile, the container
boundary needs no holes, and the same deployment serves a user who is nowhere
near it.

**Why not have the browser reorganize files directly.** The File System Access
API can write to a user-picked directory (`showDirectoryPicker()` plus
`FileSystemFileHandle.move()`), but only on Chromium desktop — Firefox has no
directory picker, Safari is read-only, and mobile has neither. It is also
**secure-context only**, so on the normal LAN address `http://<jetson>:8000` the
whole API is `undefined` — the same wall `crypto.subtle` hits above, and a fatal
one here because there is no pure-JS fallback for writing to a disk. Stage 2 is a
CLI instead (§12), which works on every OS and has no browser to disagree with.

### 3.2c The upload folder list, and deleting a folder

The library is organised as a **list of folders** — one per distinct
`uploads.root_label` — persisted in the database, so it survives restarts (the
browser file picker cannot remember a selection). `/upload` shows every folder
with its live photo count; a directory-only picker adds a new one and uploads its
photos into the library.

**Deleting a folder removes it from the LIBRARY, never from disk.** The source
folder on the user's machine is a *source* and is never read, moved, or deleted by
the app (§3.2b) — deletion only removes the library's uploaded copies and their
derived state. It removes: the folder's `uploads` rows, every photo sourced only
from that folder, and *all* of each deleted photo's metadata — `photo_vec`,
`caption_vec`, tags, facets, FTS row, jobs, group memberships, thumbnails, and the
stored original. A photo whose identical bytes still belong to another folder is
kept; only that folder's `photo_sources` path is dropped.

A memory (§11) whose photos all lived in the deleted folder is left with no
`group_photos`; the cascade prunes that now-empty `groups(kind='memory')` row and
its `memory_fts` entry, so a memory the delete emptied disappears from Organize and
chat search in lockstep — never lingering as a coverless, crash-prone orphan.

**Deletion runs through an outbox + worker**, never inline. `POST
/upload/folder/delete` records the intent in `folder_deletions` and returns
immediately; the folder shows "deleting…". The worker drains the outbox each pass
(`process_folder_deletions`) and does the cascade, so a delete is reliable across
restarts and never blocks the UI. This is the Outbox pattern.

### 3.3 Scale

- Collection: 1,000-5,000 photos for the first user.
- First upload of 5,000 photos is a background transfer measured in tens of
  minutes; the first full index may run overnight. Both must be resumable and
  observable.
- The cloud never modifies an uploaded original. Only `ivms777-sync` writes to
  your disk, only after you confirm a plan, and every operation it performs is
  journaled and reversible (section 12).

## 4. Models

The app runs **four model slots** (§4.1), all inside the one `models` service
(§5.1), loaded once — never in `app`/`worker`. **The exact slot→model→backend
table, the catalog, per-profile wiring, download mechanics, and the settled
caption-embedding decision live in [`docs/models.md`](models.md).** The shipping
defaults:

- **SigLIP 2** does image + text embeddings and zero-shot tags, in a `TorchWorker`
  child of the models service (MPS on mac, CUDA on jetson) — neither `llama-server`
  nor vLLM exposes an image-embedding endpoint, so it cannot move behind them.
- **One `gemma4-E2B` GGUF on `llama-server`** does **both** captioning (vision) and
  planner/chat (text) on mac/jetson — one model, one process, on the GPU (§3.1).
  `cloud` keeps Qwen2.5 (vLLM). Prompts are a per-model template registry, so
  adding a model is a template, not pipeline surgery.
- **A dedicated text embedder (`nomic-embed-text`)** embeds caption text for §9
  similarity, NOT SigLIP — SigLIP's text tower is trained image↔text, so text↔text
  has no separation (measured). The `caption_vec` it produces is a **top-k KNN**
  retrieval index, never a fixed cosine floor. Full rationale + benchmarks:
  [`docs/models.md`](models.md).

Gemma 4 is used over the dominated Gemma 3. The default caption model is chosen by
a **bake-off** on real photos, not published benchmarks — the winner is a config
default, not code.

### 4.1 Model slots — the catalog, and switching a model while the app runs

The models above are not hardcoded: they are the **defaults of six slots**, and the
user picks a different model per slot from the settings popup (§13) while the app is
running. A slot is the unit everything else names — the registry, the governor's
budget, the scheduler's "needs" list, and the ingest stages a switch invalidates.

| Slot | Default | Role | Switching invalidates |
|---|---|---|---|
| `image_embed` | SigLIP 2 `so400m-patch14-384` | image + text vectors, zero-shot tags | `embed` → `taxonomy` |
| `text_embed` | `nomic-embed-text-v1.5` | caption-meaning vectors (§9) | `caption_embed` |
| `caption` | `gemma4-E2B` GGUF (vision) | the caption sentence | `caption` → `caption_embed` |
| `planner` | `gemma4-E2B` GGUF (text) | query planner + chat | nothing (no stored output) |
| `voice` | Kokoro-82M (en) · Silero v4_ru (ru) | the narration read over a memory film (§11.1) | nothing (no stored output) |
| `music` | MusicGen **medium** on mac · small on jetson/cloud | the score generated per memory film (§11.1) | nothing (no stored output) |

**`off` is a real catalog entry, not a missing one.** `voice` and `music` are the two
**optional** slots, so the catalog carries an `off` entry (`source=NoSource()`,
`cost_mb=0`) for each. Selecting it unregisters the residency unit entirely, so the
governor never budgets for it and the settings popup shows it as always "on disk".
The feed then degrades honestly for MUSIC only: no music means the film plays silent.
**Narration is not optional** — a film whose voice cannot be produced FAILS (§11.1)
rather than playing mute. `music` is `off` **by default on jetson**
— its resident cost is an estimate, and an unmeasured ~3 GB child on a board with
~4.3 GB free must be opted into, never defaulted on (§8.1).

**A voice entry declares which languages it can speak.** `voice` is the one slot whose
usable set depends on something outside the slot — the film's language (§11.1). So a
`voice` entry carries `languages` (`("en",)` for Kokoro, `("ru",)` for Silero),
`POST /speak` takes the language, and it answers **503** when the selected voice cannot
speak it. Kokoro ships nine languages upstream but **not Russian**, which is exactly
why a second entry exists rather than a bigger one — and why the settings popup states
each voice's languages.

`caption` and `planner` are **separately selectable but share one `llama-server`
child** — one process, one port. If they resolve to different GGUFs the child
restarts when the role changes, which is exactly the cost shape the existing
text↔vision restart already has (§3.1); pointing both at one GGUF (the default)
keeps that restart to the projector alone.

**The catalog is data, in the repo** (`models/catalog.py`, no torch import). One
entry per selectable model, carrying everything every consumer needs so nothing
downstream has to know model names:

- `source` — the Hugging Face repo id, the GGUF repo + file (+ mmproj file), or
  `NoSource()` for the `off` entry of an optional slot (nothing to fetch, nothing
  to load, always "on disk").
- `cost_mb` — the resident cost the governor budgets against (§8.1), and whether
  that figure is **measured on this board or an estimate**. An estimate is shown
  as unverified in the UI, because an over-estimate can make a model unloadable
  and an under-estimate OOMs the box (§8.1).
- `dim` — the embedding width (embedder slots only). It decides whether a switch
  must rebuild `photo_vec` (§6).
- `preprocess` — `{input_px, resample, mode}` for image embedders. `mode: squash`
  is today's aspect-ignoring stretch to a fixed square; `mode: native` (NaFlex,
  and any native-resolution encoder) means fit-inside `input_px` with the aspect
  ratio kept. **The client resizes by this spec, never by a constant** — see §5.1's
  `/embed/spec`.
- `prompt_template` — the key into `inference/prompts.py` (LLM slots only), so
  adding a model stays "add a template", not pipeline surgery.

Only catalog entries are selectable. A free-form model id is deliberately not
offered: `cost_mb`, `dim`, and `preprocess` cannot be guessed at runtime, and each
of the three breaks something different when wrong.

**A switch is a re-index, and says so before it runs.** The selection lives in
`app_settings` (§6), owned by `app`; resolution is **stored choice → env override →
profile default**. Confirming a switch, in one transaction:

1. writes the new slot choice,
2. rebuilds `photo_vec` when `image_embed`'s `dim` differs from the live table
   (vectors from two encoders are not comparable — the old ones are dropped, not
   migrated),
3. calls `jobs.reprocess()` over the slot's invalidated stage range (table above),

then pushes the new slots to the `models` service. The `worker` drains the requeued
stages exactly like any other reprocess, with the same per-stage progress rows (§8);
search and "similar" run against a thinning index until it finishes, which is why the
popup states the photo count before it asks for confirmation.

**Every vector records the slot that made it.** The `embed` stage stamps the
**resolved** `image_embed` catalog key on `photos.embedding_model` (§6) — the column
the photo page shows ("Embedded for semantic search (…)", §13). The env override is
not the answer, and neither is a hardcoded id: reading `embed_model_name` directly
made every photo claim `siglip2-so400m-patch14-384` regardless of the selection, the
same bug the chat header had. A stamp that is **not** a catalog key is therefore a
legacy one, and the next drain pass relabels it to the current slot rather than
re-embedding a library to correct a string — the vectors were always the slot's,
since only the `models` service ever picks the model (§5.1).

**No model name is ever snapshotted — every one is read from the slot at the moment
it is used.** The rule above is not about one column; it holds for every name that is
stored, sent, or shown:

- `photos.caption_model` is the **resolved** `caption` key, reported by the service
  with the caption it produced, and the same entry's `prompt_template` is what
  builds that request — so a switched-in model is never given another model's prompt.
- the caption-vector stage (§9) embeds under the resolved `text_embed` key, whose
  family picks the task prefix — a stale name prefixes documents for an encoder that
  is not loaded, which silently degrades every caption vector.
- the resource bar (§13) and the chat header name the resolved `planner` key.

The `models` service is where this is easiest to get wrong: it builds its
sub-backends once, at boot, on the profile defaults, and `app` pushes the stored
selection only afterwards. So each sub-backend holds a **provider** of its slot entry
and re-reads it per request; a captured entry means every caption is stamped
`gemma4-E2B` and prompted as gemma no matter what the user selected.

The `models` service holds no database, so it cannot read the choice itself: it
starts on the profile defaults and reports **which model each slot currently holds**
(on `GET /resources` and `GET /models`, alongside a `generation` counter that moves
on every switch). `app` compares those reported slots with its own resolved selection
on the resource-bar poll it already makes (§13) and re-pushes when they differ — so a
`models` restart converges back to the user's selection without a coordinating store,
and no second polling loop exists. The `generation` is what invalidates the caller's
cached `/embed/spec` (§5.1).

## 5. Architecture

Three containers and one SQLite file in the cloud. One standalone binary on your
machine.

This diagram is the **canonical architecture picture** — keep it in step with the
components and flows described in this section (CLAUDE.md § "docs/design.md is the
source of truth").

```mermaid
flowchart TB
    subgraph machine["Your machine"]
        disk[("Photos on disk<br/>sources — the app never writes here")]
        browser["Browser<br/>/upload · /library · /photo · /organize · /feed · /chat"]
        sync["ivms777-sync CLI<br/>apply · undo · verify<br/>reads the downloaded plan document"]
    end

    subgraph gpu["The GPU box · docker compose (jetson/cloud) · host processes (mac)"]
        app["app · FastAPI + Jinja + HTMX<br/>UI · read queries · upload receipt · /api/plan (stage-2 plan document)<br/>NO models, NO torch — a thin client of the models service"]
        worker["worker · ingest pipeline (primary writer)<br/>facets · thumbs · taxonomy · caption · memories · feed audio · deletions<br/>NO models, NO torch — a thin client of the models service"]
        models["models · THE ONE INFERENCE GATEWAY (§5.1)<br/>the only process that imports torch/transformers AND the only client of the inference server<br/>hosts the image/text embedder in KILLABLE TorchWorker children (parent imports no torch) on the GPU — mps (mac) / cuda (jetson, cloud), NEVER cpu (§3.1) · memory governor swaps embedder↔LLM under budget (§8.1)<br/>owns the model slots + downloads (§4.1)<br/>HTTP: /embed/image · /embed/text · /embed/spec · /tag · /caption · /speak · /music · /plan · /chat · /resources · /models/*"]
        infer["inference · llama.cpp llama-server (mac/jetson) | vLLM (cloud)<br/>ONE gemma4-E2B GGUF — text (planner/chat) AND vision (caption), on the GPU<br/>reached ONLY by the models service · (host on mac, container on jetson/cloud)"]
        db[("SQLite WAL<br/>sqlite-vec + FTS5 · named volume")]
        store[("Storage<br/>originals + thumbnails + feed audio")]
    end

    disk -.->|user selects| browser
    browser -->|"upload: probe hashes + send new bytes"| app
    browser -->|"read UI · SSE chat"| app
    browser -->|"fullscreen memory film: Edit JSON + generated audio · WebGL (NO model)"| app
    browser -->|"Organize → confirm → download ivms777-plan.json"| app
    sync -->|"verify: GET /api/plan"| app
    disk -.->|"you save the plan into the folder"| sync
    sync -->|"apply / undo — the ONLY writer to disk"| disk

    app <-->|"reads · small short writes"| db
    worker <-->|"owns writes"| db
    app -->|"read originals / thumbnails"| store
    worker -->|"write originals / thumbnails"| store

    app -->|"ALL inference: query embed · planner · chat (HTTP)"| models
    app -->|"slot switch + downloads: PUT /models/slots · POST /models/download (§4.1)"| models
    worker -->|"ALL inference: embed · zero-shot tags · caption (HTTP)"| models
    models -->|"gemma4-E2B: planner/chat + caption/vision (OpenAI /v1)"| infer
    models -->|"fetch weights on demand (HF / GGUF) → model cache volume"| hf[("Hugging Face<br/>weights, fetched once per model")]

    classDef store fill:#eef2ff,stroke:#8899cc,color:#111;
    class db,store,disk,hf store;
```

`app` serves the UI, read queries, upload receipt, and the stage-2 plan endpoint.
`worker` owns the ingest pipeline and is the primary writer. Both open the same
SQLite file in WAL mode with a busy timeout — SQLite permits one writer at a
time, and `app`'s writes are small and short (recording a received upload,
accepting a group, editing vocabulary), so contention stays negligible at this
scale. If public traffic ever makes that false, the fix is Postgres, and the
repository layer is the only thing that changes.

**All model work lives in one process — the `models` service (§5.1).** `app` and
`worker` never import torch/transformers or load a model; they call the `models`
service over HTTP. This is the hard rule (CLAUDE.md § "One model process"): a model
or AI library is loaded **once**, in one process, never duplicated across
containers. It is what makes the 8 GB Jetson viable — a SigLIP loaded in both `app`
and `worker`, each with its own torch CUDA context, is exactly what exhausted the
unified memory.

**One implementation, both platforms.** The `models` service is the same code on
mac and jetson; only its *internals* switch by profile: SigLIP and the nomic
caption-text embedder run on **MPS** on mac and **CUDA** on jetson — always a GPU,
never the CPU (§3.1) — and captioning is the same OpenAI `/v1` call to
`llama-server` on both (host on mac, container on jetson). The service picks these
from config; `app`/`worker` and the whole request flow are identical across
platforms.

`ivms777-sync` is not part of the deployment. It executes a plan document the
browser downloaded — `apply` needs no network at all, and only `verify` calls
`app` — and it is the only component that ever writes to your disk.

Vector search uses `sqlite-vec`, which ships prebuilt wheels for arm64 and
x86_64, so the same code works on Mac, Jetson, and cloud. At 5,000 photos a
brute-force scan would also be fine; `sqlite-vec` is there so growth to 100k+
needs no new component.

### 5.1 The `models` service — the one inference gateway

**Every model, LLM, embedder, and heavy AI library lives in exactly one service:
the `models` service.** It is the only service that touches `torch`/`transformers`,
the only one that loads SigLIP or the caption-text embedder, and the only client of
the inference server (`llama-server` / vLLM). `app`, `worker`, and the CLI hold
**no models and no torch** — they are thin HTTP clients of this service (CLAUDE.md
§ "One model process"). Loading the same model in two processes is forbidden; on
the 8 GB unified-memory Jetson a duplicated SigLIP plus two torch CUDA contexts is
exactly what exhausted memory.

**The service owns three supervised children, and its own process holds no model.**
gemma is the `llama-server` child (§3.1); SigLIP and nomic are one `TorchWorker`
child each (§8.1, `modelsvc/torch_process.py`). The FastAPI parent imports **no**
torch at all — it starts, calls, and **kills** children, because ending the process
is the only way a CUDA context's memory comes back. This is the one-model-process
rule, not an exception to it: one owner, one copy of each model, everyone else on
HTTP.

**HTTP surface** (the whole app's inference vocabulary):

- `POST /embed/image` — SigLIP image embeddings (ingest `embed`, "similar"). The
  caller (`RemoteEmbedder`) resizes each image to SigLIP's **384×384** input,
  bilinear, before the PNG encode. That is exactly what the model's processor
  does to any input it is given (`do_resize`, `size: 384×384`, `resample:
  BILINEAR`), so the vectors are bit-identical — but it is the difference
  between a working ingest and a stalled one: PNG-encoding a full-resolution
  original cost 5.7 s of CPU and a 15 MB base64 body per photo on the Jetson,
  against ~10 ms of GPU, pinning the embed stage at 0.32 img/s with the GPU idle
  96 % of the time. Only valid for a **fixed-resolution** SigLIP checkpoint —
  a NaFlex variant consumes native resolution and would lose signal here.
- `POST /embed/text` — SigLIP **text** embeddings for search/chat query vectors
  (same joint space as the image vectors).
- `POST /tag` — SigLIP zero-shot scoring against `vocab.yaml` (ingest `taxonomy`).
- `GET /embed/spec` — the image embedder's **calibration and its preprocessing
  contract**: `logit_scale`, `logit_bias`, and the slot's `preprocess`
  (`input_px`, `resample`, `mode`) plus the slots `generation` (§4.1).
  `RemoteEmbedder` fetches and caches it, so taxonomy scoring (§9) stays
  client-side *and* the caller's pre-resize follows the model instead of a
  hardcoded 384. It re-fetches when the generation moves. (It replaces the
  calibration-only `GET /embed/calibration`.)
- `POST /caption` — a caption sentence (title/description) for one image (OpenAI
  `/v1` call to `llama-server` with the image as an `image_url`, §4). It returns
  **no tags** — tags are SigLIP-only (§7).
- `POST /speak` — `{text, language, voice}` → `{audio (base64 WAV), sample_rate,
  model}`: one narration clip for a memory film (§11.1), from the `voice` residency
  unit. `Priority.BATCH` — a film is prepared offline and must never preempt a search
  or a chat turn. **503** when the `voice` slot holds `off`, **or** when no offered
  voice lists `language` in its catalog `languages` (§4.1). Any other synthesis failure
  is a **500 carrying the model's own message**, because that reason is shown to the
  user: a film with no speech is a FAILED film (§11.1), so the error has to be
  readable, never swallowed into "no voice".
- `POST /music` — `{sections | prompt, seconds, seed, temperature, guidance}` →
  `{audio (base64 WAV), sample_rate, model, segments}`: the whole SCORE for a memory
  film in one call (§11.1). `sections` is `[{brief, seconds}]` in running order, and
  each one is generated as a **continuation** of the previous one's tail, so the reply
  is a single piece that changes rather than several clips; `segments` says where each
  section actually landed, because a token count never divides exactly into the seconds
  asked for. `prompt` is the one-section form. `temperature`/`guidance` are the
  decoding controls — left unset they are MusicGen's own defaults, which is one setting
  for every memory. Old shape, kept for reference: `{prompt, seconds, seed}` → `{audio,
  sample_rate, model}` — one clip, looped by the player, which is what made every film
  sound the same. Served from the `music` residency unit
  (MusicGen small, 32 kHz mono, generated autoregressively — the caller's `seconds` is
  clamped to a 30 s COST cap, not a model limit). `Priority.BATCH`. **503** when the
  `music` slot holds `off`.
  A 503 from either is not a failure: it is the documented "the user turned this
  off" answer, and the feed is still prepared without that track.
- `POST /plan`, `POST /chat` — text generation (query planner, chat answers).
- `GET /text/context` — how many tokens the text server accepts, read from the
  RUNNING server (llama.cpp `/props`), falling back to the configured `llm_ctx` when
  it cannot be asked. Callers that assemble a large prompt cut it to fit (§11.1);
  the setting alone will not do, because it disagrees with the server exactly when it
  matters — after a `-c` change and before a restart.
- `POST /text/embed` — caption-meaning text embeddings (§9) from the **dedicated
  text embedder `nomic-embed-text-v1.5`**, in its own worker child (`TextBackend`,
  `embedding/text_embedder.py`) — NOT `/embed/text` (that is SigLIP, image↔text
  only). Consumed as top-k KNN (§10).
- `GET /models/catalog`, `PUT /models/slots`, `POST /models/download` — the slot
  control surface (§4.1). The catalog response is the repo catalog joined with
  live per-model state: the current selection, whether the weights are already on
  disk, and a download's bytes-so-far. `PUT` swaps a slot: it evicts the outgoing
  model (killing its child, §8.1) and re-registers the slot from the new catalog
  entry — the next op loads it, nothing loads eagerly. `POST /models/download`
  fetches weights in a background thread (HF snapshot / GGUF stream into the same
  cache the entrypoint and `make llama-mac` use), never inside the request.
- `GET /resources` — **only what the service alone knows**: which models are
  resident, and the current in-flight op. It reports **no machine metrics**.
  RAM/CPU/GPU-load/temperature are host-wide numbers that any process can read
  from `psutil` and the kernel's own sysfs counters — reading them is not
  "touching the GPU" in the sense of the one-model-process rule (§5.1), it needs
  no CUDA context, no driver library and no `runtime: nvidia`. So `app` reads
  them **itself**, locally, and the bar keeps showing them even when the `models`
  service is down or still starting (§13).

**Backends, chosen by profile inside the service (one implementation, both
platforms):**

- **SigLIP** in a `TorchWorker` child — `embed_device=mps` on mac, `cuda` on
  jetson/cloud; never `cpu` (§3.1). Image **bytes** cross the worker pipe; the
  child does the decode.
- **Captioning** — an OpenAI `/v1/chat/completions` call to `llama-server`
  (mac/jetson) / vLLM (cloud); the same gemma4-E2B GGUF that answers text also does
  vision on the GPU (§3.1/§4). No in-process caption VLM.
- **Text** (`/plan`, `/chat`) — the same OpenAI-compatible inference server, via a
  shared `OpenAICompatClient`. **Caption-meaning text embeddings** run in their own
  `TorchWorker` child (`nomic`, `embedding/text_embedder.py`) on mac/jetson because
  `llama-server` has no embedding endpoint; cloud (vLLM) keeps the OpenAI
  `/embeddings` path and has no worker.

**Residency is coordinated in-process here** — there is no cross-process DB lease.
A `MemoryGovernor` (`modelsvc/governor.py`, driven by the `Scheduler`) makes the
models an op needs resident, evicting non-needed, non-pinned residents (LRU) when
either the `budget_mb` ceiling or the measured free RAM says they will not fit. On
the 8 GB Jetson the gemma child (~5.0 GB in `llama-server`) and SigLIP (~1.6 GB) +
nomic (~0.3 GB, in this process) do **not** fit together, so the governor **swaps**:
it evicts SigLIP to load gemma for a caption/chat and reloads it for the next embed.
Everything is local, so the budget is real and no second process holds a hidden copy.

This **retires the earlier cross-process model lease** (`model_lease`) and the
two-process `ModelCoordinator`: with models in one process, residency is an
in-process concern (§8.1, `modelsvc/governor.py`), and `app`/`worker` need no
coordination because they hold nothing to coordinate — they reach every model over
HTTP and load nothing themselves.

## 6. Data model

The library is one SQLite file. **The full DDL — every table and index, the
`sqlite-vec` (`photo_vec`) and FTS5 (`photo_fts`) virtual tables — lives in
[`docs/data-model.md`](data-model.md).** The shape that matters for the design:

- **`photos`** — one row per distinct image, identified by its `content_hash`
  (sha256 of the bytes). All derived state hangs off it: EXIF, caption,
  `caption_vec`, and the SigLIP vector in the `photo_vec` sqlite-vec table.
- **`photo_sources`** — every local path the bytes arrived from (many-to-one); more
  than one row is a duplicate on disk (§6.1).
- **`photo_facets`** — EXIF-derived facets in their own table, so a *fact* is never
  diluted by a model *guess* (§6.2). **`photo_tags`** holds the model-derived tags,
  each with a `score` (0..1) and a `source`, so the UI can show why a tag is present.
- **`jobs`** — one row per (photo, stage): the resumable ingest queue (§8).
- **`app_settings`** — one key/value row per owner-level setting the UI can
  change. It holds the six model-slot choices (§4.1) and **`feed_language`** — `en`
  (default) or `ru`, the language a film is written and narrated in (§11.1); an absent key means "use
  the env override, else the profile default", so a fresh library and an untouched
  install behave identically. `photo_vec`'s declared width follows the selected
  `image_embed` model's `dim` — a switch to a different width **drops and recreates
  the vec0 table** rather than migrating it (§4.1).
- **`groups` / `group_photos`** — a many-to-many junction backing Memories, so a
  photo may belong to any number of memories (§11).
- **`memory_presentations`** — one row per memory **per language** that has been
  prepared as a film (§11.1), keyed `(group_id, language)`: the `Edit` JSON, the
  storage keys of the two generated WAVs, and the library signature it was built
  against. `group_id` cascades from `groups`, so §11's atomic memory swap drops every
  stale presentation in every language for free — a recycled group id can never
  inherit a previous memory's film. A presentation is stale exactly when its stored
  signature no longer matches the library's.
- **`uploads`**, **`chat_sessions` / `chat_messages`** — the folder list (§3.2c) and
  the persisted chat transcript (§10).

`tags` is a shared vocabulary with no `owner_id`; every user-scoped query filters on
`owner_id`.

## 6.1 Exact duplicates

The same image bytes often sit in several folders under different names. Because
a photo is identified by its `content_hash`, this needs no detection pass at
all: the second copy simply adds a row to `photo_sources` against the photo that
already exists. An image stored in five places is one `photos` row with five
sources, so it is embedded, scored, and captioned exactly once, and its bytes
are stored once.

This is what makes upload the cheaper ingest model in practice. The client
probes hashes before sending (section 3.2b), so redundant copies cost one hash
comparison each and no transfer.

There is no separate duplicates screen. A photo with more than one source wears
an `×N` badge in the grid, and its detail page lists every path the bytes were
found at, with the disk space the redundant copies occupy on **your** machine.
Reclaiming that space is stage 2's job, and only after you approve a plan
(section 12).

The earlier design detected duplicates by walking a mounted filesystem, which
required a two-pass scan to tell a move from a copy. Nothing walks a filesystem
now, so that distinction disappears: the client sends what exists, and paths
that stop appearing in later uploads are simply stale sources, marked rather
than deleted.

This is exact, byte-level matching. Visually similar but not identical shots are
near-duplicates and are handled by perceptual hashing in section 11.

## 6.2 EXIF facets — external filters

Model output is a guess. EXIF is a fact. The two are kept in separate tables so
the UI can be honest about which is which, and so a filter on "shot at ISO 3200"
is never diluted by a model's opinion.

Every photo's full EXIF is stored verbatim in `exif_json`. From it, a fixed set of
**facets** is derived into `photo_facets`, each either categorical or numeric (so
ranges work) — the exact key set (Camera / Exposure / Time / Place / Image groups)
is in [`docs/data-model.md`](data-model.md#exif-facet-keys).

Facets are used four ways:

- **Filtering** — sidebar controls on `/library`, with counts. Categorical
  facets are checkboxes; numeric facets are ranges.
- **Sorting** — any numeric facet is a sort key, in addition to capture date.
- **Search** — facet predicates narrow the candidate set before semantic and
  keyword ranking (section 9).
- **Chat** — the query planner emits facet predicates, and retrieved photos
  carry their facts into the answer context, so "what lens did I use most in
  Italy?" is answered from data rather than from captions.

Facets are also a cheap quality signal the models never see: a photo at ISO
12800 with a 1/8 s shutter is probably a noisy handheld night shot, and that is
knowable without inference.

`time_of_day` uses local clock hour from EXIF, which is what a photographer
means by "evening shots". It is not recomputed from GPS and UTC.

## 7. Taxonomy

**Ten dimensions, defined in an editable `vocab.yaml`** (subject, setting, vibe,
emotion, light, season_weather, composition, palette, occasion, quality). Users add
or remove labels without touching code; re-scoring a changed dimension only re-runs
the (cheap) SigLIP stage. **The label lists, exact per-dimension sources, and the
tuning knobs are in [`docs/taxonomy.md`](taxonomy.md).**

The scoring design:

- **SigLIP is a per-dimension classifier, not an absolute detector** — raw sigmoid
  probabilities are tiny and flat, so tags are chosen by a **softmax over each
  dimension's own labels** (top label + runners-up within `select_ratio`, capped at
  `max_per_dim`), storing a real within-dimension 0..1 confidence.
- **Every label says itself in its own words.** A label is not fed to the text tower
  as its own name in a template; `vocab.yaml`'s `prompts:` block gives it one or more
  natural sentences, embedded and **averaged** (prompt ensembling), with the
  dimension's template only as a fallback. SigLIP's text tower is trained on real
  captions, so "a photo of portrait" / "a photo of top-down" / "a photo of work" cost
  real accuracy: measured against a hand-labelled sample, `subject` was right **37%**
  of the time and is now **80%** (top-3 68% → 100%), with median confidence 0.39 →
  0.76. Only the label side changes, so re-tagging a whole library is seconds.
- **The `subject` labels must be mutually exclusive**, because softmax picks exactly
  one and §9's content gate is built on it. `portrait`/`selfie`/`candid`/`group of
  people` all meant "people" and split that mass four ways — eleven family photos of
  one trip landed on four different labels and shared nothing. They are now
  `person`/`people`/`crowd` (how many), with framing moved to `composition: selfie`.
- **Captions write NO tags** (decision) — a flat-`1.0` free-text VLM guess used to
  override SigLIP's honest scores and mislabel photos. Tags come only from SigLIP +
  pixel stats (`palette`/`quality`) + EXIF.

### 7.1 Vocabulary mining

The starting vocabulary cannot anticipate one library, so a batch job mines
recurring noun phrases from captions, drops any an existing label already covers,
and offers the rest as "suggested tags". Accepting one appends it to `vocab.yaml`
and re-runs that dimension's (SigLIP-only, seconds-fast) taxonomy stage — the
vocabulary grows into the collection instead of being guessed up front. Mechanics
in [`docs/taxonomy.md`](taxonomy.md#vocabulary-mining).

## 8. Ingest pipeline

Every photo flows through a **resumable, per-stage job queue** (`jobs`, §6):
`receive` → `facets` → `thumbnail` → `embed` → `taxonomy` → `caption` →
`caption_embed`. `receive`
runs synchronously in `app`; everything after is drained by the `worker`, and
killing/restarting the container resumes exactly where it stopped. **The exact
per-stage mechanics, reprocess endpoints, and failure handling are in
[`docs/ingest-pipeline.md`](ingest-pipeline.md).**

The design decisions that matter:

- **Stages drain library-wide in order, not per photo** — every photo is embedded
  and tagged before any is captioned. On the 8 GB Jetson this is what keeps SigLIP
  and the captioner from being needed at once, and it means search + "similar" work
  across the whole collection minutes after upload, while captions fill in over
  hours. The caption's text embedding (`caption_vec`, §9) is the last stage,
  `caption_embed` — queued and counted per photo like every other stage, so it shows
  its own progress row, but **drained in one batch** (the text embedder loads once
  and embeds up to 50 captions per call), never interleaved per photo.
- **Degrade, never crash.** A drain pass runs the GPU-free stages (thumbnail, place
  facets, deletions) first, then the inference stages; if the `models` service is
  down the inference group is skipped and retried next pass, so photos still appear
  in the library and the upload receipt still returns.
- **Reprocess & self-heal.** Originals are kept, so any stage can be re-run — per
  library range or per photo — without re-uploading, and self-healing backfills
  queue any missing derived state each drain.
- **The plan gate.** `/api/plan` marks the library `complete` only when no job is
  pending/running; `ivms777-sync` refuses an incomplete plan without
  `--allow-incomplete` (§12).

### 8.1 In-process residency — the memory governor (swap under budget)

All model work lives in the one `models` service (§5.1), so residency is an
**in-process** concern, not a cross-process lease. A **`MemoryGovernor`**
(`modelsvc/governor.py`), driven by the `Scheduler` (`modelsvc/scheduler.py`),
makes the models each op needs resident and **evicts** non-needed, non-pinned
residents (LRU, oldest first) whenever the `budget_mb` ceiling *or* the measured
free RAM says they will not fit. Each unit registers a `load`/`free` pair and a
**resident cost** — the selected catalog entry's `cost_mb` (§4.1), which
`IVMS777_MODEL_COST_MB` can override per unit on a board that measures differently;
that cost must track the model's real
footprint, because the eviction guard is `free >= cost + headroom` — an
under-estimate loads a model on top of one that should have been evicted and OOMs
the box (the gemma "connection refused" bug).

**Residents are named by slot, not by model** (§4.1), so switching a model never
renames anything the governor, the scheduler or a "needs" list mentions. The
residency units are `image_embed`, `text_embed`, `llm`, `llm_vision`, `voice` and
`music`; `llm` and `llm_vision` are the `planner` and `caption` slots' two spawn
modes of one `llama-server` child on one port, so they are **mutually exclusive**:
making either resident frees the other (§3.1). A text chat never loads the vision
half. `cost_mb` comes from the selected catalog entry, not from a fixed per-name
table, so a heavier model brings its own budget figure with it.

**A slot set to `off` has no residency unit at all** (§4.1). `voice` and `music` are
optional, and "off" is not a zero-cost unit — it is an **absent** one: `SlotManager`
unregisters it, so the governor cannot be asked to load a model that does not exist,
and switching back re-registers it.

**The feed's two models are batch-only and short-lived.** `voice` and `music` are
needed only while a memory feed is being *prepared* (§11.1), never while one plays —
the player is plain HTML and loads no model at all. Both are `TorchWorker` children
on the GPU and both are ordinary governor units, so on the jetson they swap against
gemma exactly as SigLIP does. Neither cost is measured yet: `music` (MusicGen small,
~2600 MB by estimate) is therefore **`off` by default on jetson**, and both
figures must be measured on the board the way this section requires — the drop in
`psutil.virtual_memory().available` across a load — and pinned with
`IVMS777_MODEL_COST_MB='{"music": <measured>}'` before they are trusted next to gemma.

**Residency is a claim about the world, not a note to self.** gemma lives in a
separate process that can die without asking us — `llama-server` aborts (SIGABRT) the
moment a `cudaMalloc` fails, including *mid-request* at image decode. So every model
that can die behind our back registers an `alive` probe alongside `load`/`free`, and
`ModelRegistry.ensure()` **re-checks it**: a model the bookkeeping calls resident but
whose probe says dead is dropped and **loaded again**. Without that probe the registry
keeps asserting a dead gemma is resident and every caption posts to a closed port —
`connection refused`, forever, until the container is restarted. SigLIP and nomic
carry the same probe for the same reason — they are child processes too (below).

On the 8 GB Jetson gemma (~3.5 GB text / ~4.0 GB with vision, in the `llama-server`
child) and SigLIP (~3.3 GB) + nomic (~2.1 GB) do **not** fit together, so the governor
**swaps** them across the embed↔caption boundary: SigLIP is evicted to load gemma for a
caption/chat, and reloaded for the next embed. Those two figures are each child's WHOLE
footprint — its torch import and CUDA context included — because eviction kills the
child, and they are what the catalog's `cost_mb` must carry.

**A model is loaded with `device_map=<device>`, NEVER `.to(device)`.** `.to()`
materialises every weight in host RAM first, and that copy is never released: measured
on the board, SigLIP's child sat at **5128 MB** RSS loaded with `.to("cuda")` versus
**2940 MB** with `device_map` — a 2.2 GB host copy on a 7.4 GB machine. It is not
un-returned allocator arena, so `gc` + `torch.cuda.empty_cache()` + `malloc_trim`
reclaim **nothing** (all three measured, zero change); the tensors stay referenced.
`device_map` streams the checkpoint shard-by-shard straight onto the device so the host
copy never exists, and it is why `accelerate` is a dependency of the `models` extra.
This applies to every model the service loads, present and future — **with one measured
exception**: `nomic-embed-text` ships custom remote modeling code that `device_map`
splits across devices (`embed_texts` then fails with "index is on cuda:0, different
from other tensors on cpu"), so it keeps `.to()` and pays the host copy — 2.14 GB
resident instead of the 1.28 GB it would cost. That makes the caption-text embedder the
second-most expensive resident on the board, not the "~0.3 GB" earlier notes claimed.

**A torch model is evicted by ENDING ITS PROCESS, never in-process.** Freeing a torch
model in-process frees its tensors but **not** its process footprint. Measured on the
board: SigLIP takes the process to ~5.3 GB anonymous RSS, and after a full in-process
release (cache clear + `gc` + `empty_cache` + `ipc_collect`) it still holds **~2.7 GB**,
with the CUDA driver reporting only ~2.4 GB device-free —
`torch.cuda.memory_reserved()` is ~20 MB at that point, so torch has let go and the
CUDA context has not. `malloc_trim` plus clearing the cuBLAS workspaces recover ~50 MB
between them. That residue is exactly the RAM `llm_vision` (~3.9 GB measured) needs, so the
in-process swap did not work: `llama-server` aborted at load or at image decode and
every caption failed until the container was restarted.

So SigLIP and nomic each run in a **`TorchWorker` child process**
(`modelsvc/torch_process.py`, spawn — never fork), the registry's `load` is
`start()` and its `free` is `terminate()`, and `alive` is the child's liveness. The
`models` parent therefore **never imports torch** and never holds a CUDA context;
`app`/`worker` are unchanged thin HTTP clients, and no model is loaded twice — this is
the one-model-process rule, applied the same way gemma already was. Rejected, with
measurements: SigLIP on the **CPU** (residue drops to ~0.7 GB) costs **54×** throughput
(9.64 img/s GPU → 0.18 img/s CPU); `--parallel 1` on llama-server saves ~20 MB; `-ub
256` breaks vision outright (`n_ubatch >= n_tokens` is required by the non-causal image
encoder). Batch ingest runs stage-by-stage (all embeds, then all
captions), so this costs only a couple of swaps per run, not one per photo. The
scheduler also gives interactive ops (search/chat) priority over batch captioning
on the single Jetson slot, so a chat preempts a queued caption. The service reports
what is resident and the current in-flight op on `GET /resources` for the resource
bar (§13) — and nothing else; the machine metrics beside them in the bar are read
by `app` itself (§5.1).

The governor never *refuses* a load whose declared cost **fits** the budget — it
evicts SigLIP + nomic so a single gemma has the GPU to itself. But "within budget"
means `model_cost_mb[name] + headroom_mb <= ram_budget_mb`, and a model that breaks
that inequality can **never** load, on any amount of free RAM: eviction cannot help,
so the governor refuses every time. **An over-estimated cost is therefore NOT the
safe direction** — past a point it is fatal. `gemma-vision` (now `llm_vision`) was declared at 5000
against a 5000 jetson budget and a 512 MB headroom, so `5000 + 512 > 5000` raised
`InsufficientMemory` on every caption and the entire stage failed on an idle board
with 5.6 GB free. Every entry in `model_cost_mb` must satisfy that inequality for
its profile; `tests/test_config.py::test_every_model_fits_its_profile_budget`
enforces it, and the costs are **measured** (as the drop in
`psutil.virtual_memory().available`, which is what the guard reads), never guessed.
That test now runs over the whole **catalog** (§4.1): a selectable model whose
`cost_mb + headroom` exceeds a profile's budget is not offered on that profile at
all, because offering it would mean offering a slot that can never load. A catalog
entry whose cost is an estimate rather than a board measurement is labelled as such
in the settings popup — it is the one number the user is trusting. gemma always loads **fully on the GPU** (`-ngl
99`, vision projector included) — **the platform never degrades to the CPU** (§3.1).
If the all-GPU load still cannot fit (the non-evictable baseline leaves < gemma's GPU
footprint), `llama-server` aborts at load and `app` streams a plain error turn to the
chat UI; the only fix is to **shrink gemma's GPU footprint** so it fits — a smaller KV
context, a Q8 vision projector (§3.1), smaller weights — **never** a CPU offload. A
"(no answer)" bubble is a bug, not an outcome.

Implementation detail — how a sub-backend names the units it needs, the retired
caption-preemption path, and the `/resources` fields — is in
[`docs/ingest-pipeline.md`](ingest-pipeline.md#in-process-residency--the-model-conveyor).

## 9. Retrieval

Retrieval is **one core, two stages** (`search/retriever.py`, plan 12 — see
§9.2 for the full interface). **`candidates()`** is fast and has no LLM: for a
seed photo it is the image-vector KNN unioned with shared-tag, caption-meaning
and **same-moment** matches; for text it is SigLIP semantic KNN fused with FTS5
keyword search. **`refine()`** then applies the rule that governs the whole core:

**The ADR — hard filters gate, everything else scores.** EXIF facets and
explicit user/planner *facet*/*date* predicates (section 6.2) are **hard**: an
exact cut applied before ranking, and a photo that fails one is **out**, no
matter how well it would otherwise score. Every other signal — shared tags,
caption-meaning cosine, image-vector cosine, the shared moment, the
semantic/keyword fusion rank — is **soft**: a scored **contribution**. A missing
signal (no caption vector yet, no tag, no GPS) is *unavailable*, not zero, so it
is skipped for that one photo, never a kill switch. This is the rule `similar`
always followed; §10 explains why chat needed it made structural too.

EXIF time and place are the exception that proves the rule: as a *predicate* they
are a hard filter (§6.2), and as a *similarity signal* — the `moment` below — they
are soft like everything else. Same data, two roles, never confused.

What used to be described as four separate mechanisms are now candidate
generation and contributions of the one core:

- **Semantic** — query text through the SigLIP text encoder, KNN via
  `sqlite-vec`, inside `candidates()`. Handles "dogs playing in snow" with no
  matching caption.
- **Keyword** — FTS5 BM25 over captions and tag text, fused into `candidates()`
  alongside semantic. Catches proper nouns, OCR'd text, and exact words
  embeddings smear over.
- **EXIF facets** — exact filters over `photo_facets` (section 6.2), now the
  core's hard pre-filter (`_hard_filter`, §9.2): applied to the candidate list
  before any scoring, since they are cheap and exact.
- **Tag facets** — model-derived tag filters are no longer a pre-ranking SQL
  narrow; a sidebar/planner tag hint is a **soft** contribution
  (`_soft_tag_contributions`, §9.2) scored by `refine()`, never a gate.
- **Moment** — for a seed photo only, EXIF time and GPS also generate candidates
  (`search/moment.py`): photos from the same stretch of time in the same place,
  which no model-derived signal can reach. It is the only source needing neither
  an embedding nor a caption.

`/library` search calls `candidates()` directly for the fused ranking, then
narrows that list by the sidebar's EXIF/tag filters — the same
candidate-then-filter order the pre-core fusion always used, just generated by
the shared core now, with no per-click LLM latency (§9.1). `similar`, chat, and
memory call the fuller `refine(candidates())` for the graceful additive score.

**Retriever core flow.** Keep this diagram in step with `search/retriever.py`.

```mermaid
flowchart TB
    q["Query · text OR seed_photo_id<br/>+ hard_filters + soft_tags + k + floor (§9.2)"] --> gen

    subgraph gen["1 · candidates() — fast, no LLM"]
        t["text → SigLIP text→image KNN ⊕ FTS keyword, RRF-fused"]
        s["seed → image-vector KNN ∪ shared-tag ∪ caption-meaning ∪ same-moment<br/>(moment needs no embedding — it alone survives tier 0)"]
    end

    gen --> hard["2 · HARD pre-filter (_hard_filter)<br/>EXIF facets + explicit date (§6.2) — exact cut"]
    hard --> score["3 · refine(): gated evidence (search/signals.py + search/scoring.py)<br/>tags ⊕ caption cosine ⊕ image cosine ⊕ moment ⊕ fused rank<br/>each gated then weighted · per-photo graceful skip · content gate<br/>composed by noisy-OR → a true 0–1 score"]
    score --> floorstep["4 · Optional floor (caller-set)<br/>similar: similar_score_min · search: none (rank, don't cut)<br/>chat ranks separately (§10: top-k + agent)"]
    floorstep --> out["ranked [{id, score, reasons}]"]
```

The query planner (§9.1) sits **outside** this core — it is an **input
adapter** that turns free text into `hard_filters` + `soft_tags` once, then
hands the core a `Query`. It is not part of the core itself: keeping it outside
is what keeps the core LLM-free and single-pass, honouring §9.1's
interactive-latency rule.

**Similar photos** — a photo's whole character, from **seven signals** composed into
one 0–1 score, and it **degrades gracefully with the pipeline** so it is useful the
moment a photo is uploaded and richer once it is embedded, tagged and captioned. This
scorer now *is* the core's `refine()` (§9.2) — `similar_photos` is a thin wrapper over
`refine(candidates(Query(seed_photo_id=...)))` — so what follows describes the core's
seed-query scoring, not a separate mechanism.

**Every signal has exactly two numbers: a gate and a weight.** The gate is the
admission bar — below it the signal is **absent**, never a small number. The weight is
the cap — the most that signal may ever claim on its own, on one shared 0–1 scale. In
weight order: **image** (SigLIP image↔image cosine) · **subject** (the shared subject
tag) · **caption** (caption-embedding cosine) · **where** (`setting`, `occasion`) ·
**moment** (same time, same place) · **look** (`light`, `season_weather`, `palette`,
`vibe`, `composition`, `emotion`) · **quality** (a pure tiebreak). **The exact gates,
weights, and the measured distribution behind each one are in
[`docs/retrieval.md`](retrieval.md#similar-photo-scoring); `search/signals.py` is the
single place they live in code.**

**`moment` — same time, same place.** The one signal that never looks at the picture.
Two photos can share no subject, no palette and no visual resemblance and still belong
together because they were taken minutes apart in the same spot: the cake and the faces
around it, the trailhead sign and the summit. `exp(-Δt/12h) × exp(-Δd/1km)` from EXIF
time and GPS, so both must be close; **without GPS it falls back to time alone and is
capped**, because "same hour" without "same place" is a much weaker claim. It needs no
embedding at all, so a just-uploaded photo already has similar photos — graceful
degradation taken to its limit.

**Choosing a gate — the one rule: NEVER below the library's random-pair median.**
Cosines do not start at 0. Two RANDOM photos of the reference library score 0.558
(image) and 0.621 (caption). The bug this model replaced broke exactly that rule —
the caption bar was 0.60, *under* its own noise median, so 64 % of all pairs cleared it
and the caption was scoring pure chance. And because a cosine's gate sits high, a pair
sitting just above it must not score ~0 or admitting it buys nothing: **crossing a gate
is immediately worth half the signal's weight**, rising to full at 1.0. Tags need no
such ramp — they are already softmax probabilities — and are damped by `idf` instead,
which is the tag world's equivalent noise floor.

**Content gate.** A candidate is "similar" ONLY if it shares a **content** signal: a
visual near-dup, a confident shared `subject`, a caption that means the same, or the
same `moment`. Style and scene facets **only rerank** — they never make two photos
similar on their own, however many of them agree. Two photos both shot top-down in cool
overcast light are not "the same thing". **The gate IS the content bar**: a signal that
cleared its gate is real enough to qualify, and one that did not is absent.

**Composition — a noisy-OR, not a sum.** `score = 1 − Π(1 − evidence)`. Each signal
takes a share of what is still unclaimed, which buys three things a decayed sum did
not: the score is a true **0–1** the UI can print as a percentage instead of an
unbounded number chased by a magic floor; one strong signal dominates for free, with no
decay constant to tune; and weak signals **mathematically cannot stack** into a false
positive.

**Score floor.** Below `similar_score_min` a candidate is not shown at all, so a photo
with nothing genuinely like it in the library returns **fewer results, or none** —
never a full strip of the best of a bad set.

**Two gate profiles — `STRICT` and `LOOSE`.** The strip shows `STRICT`: every gate in
the top few per cent of its distribution, so a result is there because something real
matched. When `STRICT` does not fill the strip, a **"Show more"** button offers
`LOOSE` — the same signals at the same weights, admitted at lower bars. Only gates
relax; changing weights would reshuffle results that already rank below the strict
ones and make the two passes incomparable. `LOOSE` still obeys the noise-floor rule
above, which is what makes "looser" honest rather than invented.

The loose results are **appended, never merged**, so a weak match can never outrank a
real one, and they exclude everything the strict pass already showed. They are dimmed
and captioned so the difference is visible. This is where a *moment-only* pair belongs:
measured on the library, a tire close-up matched a plastic container photographed the
same minute — the same outing, an unrelated object. Worth offering, not worth putting
in the default strip.

Each result carries its **top-3 reasons** with a match % (§13), ordered by **what
actually drove the match** — not by the biggest percentage. `quality: sharp` agrees at
100 % on almost every pair and drives ~0.01; sorting by percentage put it at the top of
403 result cards, and headlined two unrelated photos with `light: low light 69 %`. A
big number is not a big reason.

Pure image-vector KNN as the *primary* signal was rejected (a dog on a rooftop returned
other rooftops); so was an LLM reranker for this interactive path — it reintroduces
per-click latency §9.1 forbids.

**Similar-photo flow.** Degrades with the pipeline; the content gate is mandatory.

```mermaid
flowchart TB
    p["A photo"] --> sig{"What signals exist?"}
    sig -->|"EXIF only"| m["moment · same time &amp; place<br/>exp(-Δt/12h) × exp(-Δd/1km) ≥ gate<br/>(no GPS → time alone, capped)"]
    sig -->|"+ embedding"| v["image cosine ≥ gate<br/>(gate ≫ the 0.558 random-pair median)"]
    sig -->|"+ taxonomy"| t["shared tags · 4 tiers<br/>subject · where · look · quality<br/>tier_weight × agreement × idf"]
    sig -->|"+ caption"| c["caption meaning<br/>caption-vec cosine ≥ gate<br/>(gate ≫ the 0.621 random-pair median)"]

    m --> gate{"CONTENT GATE — shares a content signal?<br/>image · subject · caption · moment<br/>(clearing the gate IS the content bar)"}
    v --> gate
    t --> gate
    c --> gate
    gate -->|no| drop["not similar<br/>(where/look/quality only RERANK,<br/>they never qualify)"]
    gate -->|yes| score["evidence = weight × strength<br/>score = 1 − Π(1 − evidence) · noisy-OR<br/>→ true 0–1; weak signals cannot stack"]
    score --> floor{"score ≥ score_min?"}
    floor -->|no| drop
    floor -->|yes| reasons["top-3 reasons, strongest DRIVER first<br/>+ match % · overlaid on each thumbnail (§13)"]
    drop -.->|"STRICT strip not full →<br/>'Show more' re-runs with LOOSE gates"| sig
    reasons --> strip["STRICT strip (k=12)<br/>+ LOOSE tail APPENDED below, dimmed,<br/>minus everything strict already showed"]
```

The dotted edge is the only way `LOOSE` is ever reached: the user asks for it, on a
strip the strict pass could not fill.

### 9.1 Query planner

The planner model (§4) converts a natural-language query into a `QuerySpec` in one
call — a `semantic` string plus `date_from`/`date_to`, categorical `tags`, and
`facets` (numeric `gte`/`lte` or categorical lists) that map directly onto
`photo_facets`. Its predicates are materialized into the same filter params the
sidebar uses, so the parsed filters show as **removable chips**; removing one drops a
predicate and re-runs the ordinary filtered search, and the planner does not run
again until a new query is typed. **Example spec + the exact param mapping:
[`docs/retrieval.md`](retrieval.md#query-planner-output).**

A single structured-output call, not an agent loop. A multi-step tool-calling
agent would cost seconds per step to query one SQLite table, which is a bad
trade. This applies to *interactive* retrieval only. The Memories organizer
(section 11) does run an agent loop, because it is a batch, offline job whose
per-step latency is paid once at build time, not on every keystroke.

**Chat is the one interactive exception (§10).** The user accepted the latency of
an agent loop for chat specifically — a wrong answer there is worse than a slow
one — so chat may run a bounded multi-round retrieval loop. `/library` search
stays a single call; chat does not.

The planner is strictly an enhancement. If it fails, times out, or returns
invalid JSON, the raw query goes straight to semantic + keyword fusion. The UI
shows the parsed filters as removable chips, so the interpretation is always
visible and correctable.

### 9.2 The retriever core

`search/retriever.py` is the **one** retrieval pipeline (plan 12). Every
consumer — `/library` search, `/photo` similar, chat, memory — is a caller of
it; nothing else in the codebase scores, fuses, narrows, or floors photos.

**The `Query` shape, the `candidates()`/`refine()` signatures, and the async-paint
detail are in [`docs/retrieval.md`](retrieval.md).**

**Two tiers, one core.** The **fast tier** (`/library` search, `/photo` similar)
calls the core directly — no agent, no per-click latency (§9.1). The **agentic-RAG
tier** (chat §10, memory §11) layers judgement — verify, curate, decide membership —
*on top* of the same `candidates()`/`refine()`, never a second retrieval pass.

```mermaid
flowchart TB
    subgraph fast["Interactive · no LLM · single pass (§9.1)"]
        lib["/library search"] --> core
        sim["/photo similar<br/>(similar_photos = thin wrapper)"] --> core
    end
    subgraph agentic["Agentic wrappers · LLM, latency-tolerant"]
        chat["/chat<br/>route → SigLIP library / memory search OR general answer"] --> core
        mem["Memories<br/>theme discovery + event context tools"] --> core
    end
    core["search/retriever.py · graceful additive core"]
    core --> stores[("photo_vec · caption_vec · photo_tags · photo_facets · FTS")]
```

**The single-pipeline invariant.** If a code review ever finds a second place
that scores, fuses, narrows, or floors photos, that is a bug against this
design — `candidates()`/`refine()` are the only two stages, everywhere.

`/photo` already paints instantly with the similar strip loading async; splitting
that fragment into KNN-paint-then-refine is still future — see
[`docs/retrieval.md`](retrieval.md#async-paint-photo).

## 10. Ask-your-library chat

Retrieval is **agentic RAG**, not a one-shot dump. The old path fused the top-30
neighbours and force-fed them to the model, which then invented matches ("a dog
on the dashboard" with no dog) because a semantic KNN always returns *k*
neighbours — there was no "nothing matches". The path is now precise, cheapest
layer first — and the deterministic questions never touch the model at all:

Cheapest layer first, and the deterministic questions never touch the model:

0. **Direct-DB layer.** Everything the DB can answer *unambiguously* — the library
   total, a subject FTS count, the memory count, a month/year span, and **showing a
   memory / all memories** — is answered straight from SQLite by `direct_answer`,
   with **no model call**. Each matcher is **conservative**: it fires only when
   confident, else falls through to the agent, so an odd phrasing degrades to the
   agent, never to a confidently-wrong answer (the boundary is pinned by a routing
   matrix of adversarial phrasings). A memory-show also renders the Organize card(s),
   linked `ctx=chat-memory:<key>` so opening one pages within the memory and closes
   back to the conversation (§13.1).
1. **Route (plan 17).** One schema-constrained planner call (temp 0) classifies the
   message into `search_library`, `search_memories`, or `none` (general
   knowledge/chit-chat). Failure falls back to `none`.
2. **Run the tool.** `search_library` → **SigLIP image↔text** (`search_photos`, §9.2:
   query text vs each photo's *image* vector — NOT captions);
   `search_memories` → name/theme match, pure SQL; `none` → no retrieval at all.
3. **Answer (streamed, temp 0).** A tool result grounds the model on ONLY those
   photos/memories (id, date, caption, top tags, EXIF), citing `[photo:ID]`; the
   prompt forbids inventing a subject and reports none when nothing matches
   (honest-empty). `none` answers from the model's own knowledge — chat is a general
   assistant, not photo-limited.

#### Never wake SigLIP until the question is known to be about photos

SigLIP and gemma cannot be co-resident on the Jetson — 3400 + 3800 against a 5000 MB
budget (§8.1) — so loading one **evicts** the other, and bringing gemma back respawns
`llama-server` and reloads ~3 GB: **~9 s, measured on the board, plus a lost prompt
cache**. gemma is also the model most likely to be **already resident**, since every
turn ends with it. That asymmetry decides the order:

- **Guardrails ON — retrieve FIRST.** An off-topic message is refused, so anything
  answered is about the library: the bet that this question needs photos always wins.
  The turn is `siglip → gemma` — **one** swap — and the pool retrieved up front is
  reused by `search_library` instead of embedding a second time.
- **Guardrails OFF (default) — gemma FIRST.** Only the model can tell whether the
  message is about the library, so it decides before anything is loaded. *"How many
  planets are in the solar system"* then costs **no swap at all**; a photo question
  pays one only when it genuinely needs photos.

Speculatively priming SigLIP on every message is therefore **wrong** with guardrails
off: it evicts a resident gemma and reloads it after, buying two swaps for a search
that never happens. Retrieval-before-generation is an optimisation for the case where
retrieval is certain — not a default.

The same rule governs the fully-agentic loop: `agentic_gather(prime=…)` is passed
`prefs.guardrails`, and its first instruction is to answer with no tool unless the
message is about the user's own photos/memories. `search` is reserved for "show me my
photos of X" — never for a count, never for general knowledge.

**Query text is lower-cased before SigLIP** (`search_photos`). Its text tower is
trained on lower-case web captions and is case-sensitive in a way that misleads:
an upper-case word reads as text *printed in* the image, so `DOG` returned documents
and ID cards while `dog` returned the dog photo at rank 1 — measured on the board.

**Two per-owner toggles (`chat_prefs`), defaults reproducing the pipeline above:**
**Direct answers** (on) — off skips the direct-DB step and runs a fully-agentic tool
loop (`count_photos` / `list_memories` / `count_periods` / `search`) instead; and
**Guardrails** (off) — on turns the router's `none` verdict into an on-topic gate
(fixed refusal, no generation), except app-topic questions, which are never refused.
The two are independent. **Full mechanics — matchers, tool schemas, grounding budget,
guardrail specifics — in [`docs/chat.md`](chat.md).**

**Agentic RAG flow.** Direct-DB first when *Direct answers* is on (no model); the
semantic tail degrades to plain fusion at every stage.

```mermaid
flowchart TB
    quest["User question · /chat"] --> dpref{"chat_prefs.direct_answers?"}
    dpref -->|on| direct{"Direct-DB answerable?<br/>direct_answer — total · subject FTS · memory count · memory show/list · periods<br/>conservative: declines when unsure"}
    dpref -->|off| lease2["Take CHAT lease"]
    direct -->|yes| dbans["Answer straight from SQLite<br/>NO model call · memory-show also renders the Organize card(s) (event: memory)<br/>covers link ctx=chat-memory:key → page within the memory, close → /chat"]
    direct -->|declines| gpre{"chat_prefs.guardrails ON?<br/>THE SigLIP GATE — does policy already know this is about photos?"}
    lease2 --> gpre
    gpre -->|"ON · on-topic is GUARANTEED"| prime["0 · RETRIEVE FIRST — no model call<br/>SigLIP image↔text on the question (search_photos, §9.2, lower-cased)<br/>query text vs each photo's IMAGE vector · top-k · NO captions/caption_vec<br/>the bet always wins here, so the turn is siglip → gemma = ONE swap (§8.1)"]
    prime --> route
    gpre -->|"OFF · direct on — gemma decides FIRST"| route{"Route — planner, temp 0, schema (plan 17)<br/>search_library · search_memories · none"}
    gpre -->|"OFF · direct off — gemma decides FIRST"| agentic
    route -->|none & guardrails on & NOT app-topic| refuse["Refuse — fixed GUARDRAIL_REFUSAL<br/>NO model generation · NO search · persisted<br/>is_app_topic (counts/memories/albums/…) overrides → never refused"]
    route -->|search_library| lib["SigLIP image↔text (search_photos)<br/>guardrails ON: REUSE the primed pool, never embed twice<br/>guardrails OFF: embed NOW — the first moment photos are known to be needed"]
    route -->|search_memories| mem["Memory search by name/theme (memories_for_show) · pure SQL"]
    route -->|none| gen["General answer from the model's own knowledge<br/>SigLIP NEVER LOADED — no swap at all<br/>chat is NOT photo-limited · no search"]
    agentic["Fully-agentic loop (direct OFF)<br/>prime=guardrails: pool retrieved up front ONLY under guardrails<br/>STEP 1 — about the user's OWN photos/memories? no → answer, NO tool, NO SigLIP<br/>yes → model calls REAL tools: count_photos · list_memories · count_periods · search<br/>search is the only SigLIP op, so it is last-resort: only to SHOW photos"] --> ground
    lib --> ground["Answer grounded ONLY on results/facts<br/>id·date·caption·tags·EXIF · cite [photo:id]<br/>strict: never invent · honest-empty when none match"]
    mem --> ground
    ground --> ans["Stream SSE · temp 0 · cites render as thumbnails"]
    gen --> ans
    refuse --> ans
```

**Degrade, never crash & honest-empty.** Any route/search/embed/generation failure
falls back to a plain answer (failed route → `none`; failed `search_library` → the
core's `candidates()` fusion, §9.2), so chat always answers and no ranking lives
outside the core (single-pipeline invariant). The grounding rule "never invent,
report none" means an empty search yields an honest "none", not a fabrication; the
chat view always shows its cited source thumbnails as the evidence.

**History is persisted** (`chat_sessions`/`chat_messages`, §6): `/chat` renders the
current session server-side on load, so the conversation survives navigation and
restarts; **New session** starts a fresh one. Each question is answered independently
against freshly retrieved photos — the transcript is not multi-turn model memory. The
chat route calls the small, always-loaded planner directly, so a question during
indexing does not evict the captioner.

## 11. Organize — albums by principle

The **Organize** tab groups the library into albums by a principle the user
picks from a dropdown. Most organizers recompute their albums live — the grouping
is a view over the current data, so it is always up to date, with no
accept/dismiss step and nothing persisted. **Memories** is the one exception: it
runs an agent, is far too slow to recompute per view, so it is built in the
background and stored (below).

An **organizer** is a function from the library to a list of albums. Each album
has a title, a description, a cover, and its photos. The organizers live in
`albums/` behind an `Organizer` protocol, so a new principle is one module and
one dropdown entry.

**By date.** A `grain` sub-selector picks the calendar bucket — `day`, `month`
(default), or `year`. It is a query parameter on `/organize` alongside the
organizer name and needs only EXIF, no model. Title is the period ("14 Jun 2025",
"June 2025", "2025"); description is the photo count and dominant camera — "140
photos, mostly Canon EOS R6."

Day, month, and year are the **same axis at three zooms** — a real hierarchy. There
is deliberately no "events" grouping here: time-gap clustering is a different
principle, not a zoom level of the calendar, so it never sits in this sub-selector.
It lives on only as an internal seeding step for Memories (below).

**By camera / device.** Group by EXIF camera model. Exact, trivial.

**By place.** Group photos by where they were taken and title each album with a
**real place name** — "Kyiv", "Rome", "Toronto" — from offline reverse geocoding
of the GPS, **never raw coordinates**. A bundled GeoNames dataset (`reverse_geocoder`)
resolves each point on the box, with no network, so one city is one album. Only
photos carrying GPS appear. Coordinates are a technical detail, shown only on
`/photo` (§13) and never in Organize — bare lat/long is not a place a person
recognizes. The library also filters by place: the facets stage reverse-geocodes
GPS into `place_city`/`place_country` facets (§6.2), shown as a "Place" group in
the `/library` sidebar.

**Memories.** The interesting one, and the reason the Organize tab needs the LLM.
It composes the library into named, described *memories* — "A day at Borjomi",
"Family night in Ontario" — not sets of look-alike photos.

**The governing principle: the LLM decides every membership; heuristics only make
the problem tractable.** How photos combine into a memory is a judgement call — it
needs a model that has read what is *in* the photos, not a distance formula. So no
heuristic ever decides a memory's contents. Time/place/retrieval are used *only*
to hand the agent a small, relevant working set (897 photos will never fit one
context); within and across those sets the **agent alone** decides what is one
memory, what splits, what merges, and which photos belong — including the **same
photo in several memories**.

**Two kinds of memory, and why overlap is the point.**

- **Event** — a time-and-place-bounded happening: "A day at Borjomi, 1 Dec 2023."
- **Theme** — a thread across time: "Waterfalls", "With grandma", "Autumn 2023".

A single photo naturally belongs to **one event and several themes** — the
waterfall shot is in "A day at Borjomi" *and* "Waterfalls" *and* "2023 in
Georgia". Memories are therefore **overlapping sets, not a partition**. The data
model already supports this for free: `group_photos` is a many-to-many junction, so
a photo id may appear in any number of `groups(kind='memory')` — **no schema
change**. Overlap is a feature to embrace, not a conflict to resolve.

The composition pipeline is five agent-driven passes — **pool** (cheap, bounds
context) → **compose events** (per session) → **discover themes** (RAG) →
**reconcile** → **persist** (atomic swap into `groups(kind='memory')` +
`group_photos`, re-indexing `memory_fts` so chat can find them). Every membership is
the agent's; the pooling only bounds context, and its expand tools route through the
one retriever core (§9.2). **Step-by-step mechanics, tools, and persistence are in
[`docs/memories.md`](memories.md).**

**Memory composition flow.** The agent decides every membership; heuristics only
bound context. Runs only over processed photos.

```mermaid
flowchart TB
    mrstart["Rebuild memories (manual, background thread)<br/>agent tools + embeds call the models service (§5.1)<br/>embed calls: HIGH priority — preempt an in-flight caption on jetson (§8.1)"] --> lib
    lib["Owner's PROCESSED photos<br/>caption + embedding present"] --> pool["1 · Pool (cheap, NO decisions)<br/>coarse sessions by time + ~50 km region<br/>6 h gap at home · 36 h while staying away<br/>(a week's trip is ONE session)<br/>— only to bound context"]
    pool --> events["2 · Compose events (AGENT, per session)<br/>reads summaries, decides the carve<br/>tools: similar · facets · nearby-in-time · same-subject"]
    lib --> themes["3 · Discover themes (AGENT + RAG)<br/>propose thread → retrieve candidates → curate"]
    events --> recon["4 · Reconcile (AGENT)<br/>dedupe, merge fragments, final titles/covers<br/>keeps event⇆theme overlap"]
    themes --> recon
    recon --> persist["5 · Persist (atomic swap)<br/>groups(kind='memory') + group_photos (many-to-many)<br/>+ re-index name/description → memory_fts (chat memory-show, §10)<br/>params = signature: owner count + newest updated_at"]
    persist --> ui["/organize?by=memories<br/>manual background rebuild · stale flag when signature moves"]
```

**Cost is bounded** (§9.1's batch, offline exception): agent loops are capped and the
build is signature-guarded, run only on demand. **Rebuilding is manual**, on a
background thread, one at a time; the tab flags **stale** when the library signature
(owner photo count + newest `updated_at`) moves, and a rebuild runs only after
captioning/embedding completes. Rebuild details, and the superseded heuristic seeder,
are in [`docs/memories.md`](memories.md#cost-and-rebuild).

**Organize is also the entry to stage 2.** Whatever principle is on screen can be
turned into folders on your own disk: **Organize files…** previews the tree that
principle would create, asks whether to reorganize that way, and downloads a plan
document the local CLI executes (§12). The organizer *is* the layout — nothing else
is configured. Overlapping memories are resolved at that boundary only, in the plan:
the event owns the file, themes become description-only folders. The app itself keeps
the overlap.

### 11.1 Playing a memory — the film

A memory is a story, so it is **played as a short film**, not paged as a slideshow.
`/feed/{key}` is a fullscreen presentation of one memory: the camera moves through
each photo, shots dissolve and cut into one another on a deliberate rhythm, a
generated atmospheric track plays under it, and a generated voice reads a script
written for that memory — in English or Russian.

**Generation is offline; playback loads no model.** This is the governing rule, and
it is what keeps the film viable on the 8 GB board. A background job prepares the
artifacts once per memory **per language**; the player is plain WebGL over the
thumbnails that already exist.

**One timeline, not three things playing at once.** Picture, score and voice are the
same clock. Every shot carries an **absolute `at`**, the player looks the current shot
up by the audio element's own `currentTime`, and a cut is placed on a **detected
downbeat** while a title card is placed on the **exact narration sentence** that says
it. A duration-chain drifts against audio within seconds, and a drifting cut is the
whole difference between a film and a slideshow with a soundtrack near it.

**Five stages, and two of them use a model.**

1. **Signals** (`feed/signals.py`) — read what the library already knows about each
   photo: orientation from `width`/`height`, the top-scoring label per tag dimension
   (`vibe`, `palette`, `light`, `quality`, `subject`, …), `place_city`, `shot_at`,
   and its SigLIP vector. Pure SQL.
2. **Direct** (`feed/director.py`) — turn those signals into an **`Edit`**: for each
   shot a duration, a **camera move** (scale + translation, eased), the **transition**
   into it, a **colour grade**, and whether it is a **hero** shot. Deterministic, no
   model, so the same memory always cuts the same way. The rules:
   - **the camera only ever pushes**, never pans, and never reverses between shots.
     An earlier version alternated direction and swept sideways along the photo's long
     edge; together those made the film rock like a boat for its whole runtime. The
     variation is in HOW MUCH a shot moves, never in which way. The one exception is
     the **opening shot, which pulls back** — a reveal is how a film starts, and a
     single bookend is not an oscillation;
   - a **burst** — consecutive shots inside 90 s whose image cosine clears
     `STRICT.image` (0.80, §9) — collapses into a fast montage of hard cuts, because
     six near-identical frames are one moment, not six;
   - a **hero** shot (`quality=sharp` high, `subject=person` present) is held longer
     on a slow push-in;
   - a **chapter beat** — a time gap over three hours, or a change of `place_city` —
     is a dip to black;
   - `palette` + `light` pick the **grade**;
   - pacing opens slow, tightens through the middle, and closes slow.
3. **Story** (`feed/story.py`) — ONE model call, for the things taste owns: the
   narration script, the sparse title cards, and the music prompt. It is shown the
   **same rich briefs the conductor gets** — every taxonomy dimension, the place, the
   hour, the camera — plus a **digest of the memory as a whole** (`signals.digest`):
   how many photographs, over what span, where, and the dominant occasion, mood,
   weather and light. It was previously shown only the bare vision sentence, so it
   wrote about pictures instead of about an evening; the labels are what turn "several
   individuals in front of illuminated structures" into "a snowy holiday night in
   Tbilisi". Written **directly in the target language**, so it is also the translation — the shot lines
   are never the raw `ai_title`, which is object labelling ("Spray Bottle in Hand")
   and reads as such.
4. **Audio** — narration (`POST /speak`) and the score (`POST /music`), both `BATCH`.
   **Narration is MANDATORY and music is not.** A film is a story being told; with no
   voice there is no film, only a slideshow. So a narration that cannot be written in
   the film's language, or cannot be spoken, **fails the whole prepare**: the row is
   stored `status='failed'` with the reason, the page shows that reason instead of a
   player, and nothing half-made is presented as ready. Music failing is still only a
   silent film.

   **The voice reports WHERE each sentence lands.** `POST /speak` returns `segments`
   — `[{text, start, end}]` — because only the voice child knows how long a sentence
   took to say, and the conductor cannot put a picture under a sentence it cannot
   locate. Silero gets them free from the length-limit chunking below; kokoro already
   yields per sentence.

   **The narration decides the runtime.** The film is as long as the voice plus a
   three-second tail, not as long as a sum of nominal shot holds. It used to be the
   other way round, which is exactly how the voice ended up finishing halfway through
   the pictures. The conductor's plan is then **clamped to that runtime** — the last
   shot lands exactly on it — because a model asked to fill sixty seconds will happily
   return ninety, and a film that outlives its own audio finishes in silence, which
   reads as a hang.

   **The script is sized to the PHOTOGRAPHS** (`story.narration_words`): about 14-22
   words per shot, floored at 55 and capped at 400. A fixed 200-400 words over a
   three-photo memory is a minute of speech across three pictures — a forty-second hold
   on one still, which is not a long take, it is a stall.

   **A run that is INTERRUPTED says so.** A prepare runs on a daemon thread; if the
process goes away underneath it — a code reload, a restart, a crash — nothing is written
and the page simply shows the Prepare button again, which is indistinguishable from
never having clicked it. So a `running` row is written when the run starts, and a
`running` row with no live thread is reported as "the last run was interrupted". It is
only written when there is nothing to lose: an interrupted **Regenerate** leaves the
previous film playable.

**A stored film with no `narration_key` is never played.** Rows written before
   narration became mandatory are silent, and some carry an English script under a
   Russian film; the page offers to prepare them again and says why, rather than
   playing one.

   **Long narration is split before it reaches the voice.** The script is 200-400
   words, and Silero v4_ru refuses anything past roughly a thousand symbols
   (`Model couldn't generate your text, probably it's too long`) — which silently cost
   every Russian film its voice. The Silero child therefore cuts the script at sentence
   boundaries into pieces well under that ceiling and concatenates the clips itself, so
   one WAV still crosses the boundary. Kokoro already chunks per sentence internally.

5. **Conduct** (`feed/conductor.py`) — the SECOND model call, and the one that makes
   the film one thing. It is briefed with everything the machine knows and asked for a
   single plan:

   - every photograph's **full signal set** (`ShotSignal.brief()`): the rich
     description, all ten taxonomy dimensions, place, time, camera, aperture, ISO,
     orientation — so it can answer *which picture is this sentence about?*;
   - **where every narration sentence starts and ends**, to the millisecond;
   - **where every musical downbeat falls** (`feed/beats.py`), plus the tempo;
   - the **effect library** (`feed/effects.py`), each entry describing what it looks
     like, when to reach for it and when not to, with every parameter's range explained.

   It returns, per position: which photograph plays, when it starts, how long it holds,
   which effect with which parameters, and how it arrives. **It may reorder the film** —
   showing the cat while the voice talks about the cat is the entire point, and a frozen
   order cannot do that.

   Nothing trusts the answer. `director.with_arrangement` repairs it: every photograph
   appears exactly once, times are forced monotonic, names are validated, out-of-range
   numbers are clamped, and a plan of the wrong length is dropped whole. A failed call
   falls back to an even, beat-snapped arrangement, so a film never depends on it.

**The beat grid is signal processing, not a model** (`feed/beats.py`). Spectral flux →
onset envelope → tempo by autocorrelation → a phase-fitted pulse train, in numpy, on the
clip we just generated. Cuts snap to a downbeat only within 0.35 s: sync serves the edit,
it does not overrule it.

**Nothing may JUMP at a cut.** The outgoing shot ends its push at, say, 1.12 while the
incoming one begins at 1.00 — and because a dissolve shows both at once, the eye reads
that difference as the picture snapping as the blend crosses over. So the incoming shot
does not start on its own trajectory: it starts at **exactly the outgoing shot's current
transform** — scale, offset and blur — and eases onto its own across the transition. At
`tTrans = 0` the two are identical, so there is nothing to jump; by `tTrans = 1` the new
shot owns its move. A hard `cut` is deliberately exempt: changing everything at once is
what a cut is for. Beat weights are eased for the same reason — a linear ramp announces
its own start and stop, where an eased one just becomes the next thing.

**The craft pass ROTATES through the library** (`director._default_beats`). Which bed a
shot gets suits the photograph, but *which of the suitable ones* rotates by shot index —
deterministically, so a memory always cuts the same way. Without it, a memory whose
photographs share one `light` tag received one identical bed on every shot and the film
read as "two effects": a bit of light and a bit of movement.

**A shot is an ARC, not a filter** (`effects.pack_beats`). One treatment held for eight
seconds is a screensaver, so every shot carries a **sequence of beats** —
`{effect, params, from, to}` with the times as fractions of that shot — and they
**overlap**, so one hands over to the next instead of snapping. At most `MAX_ACTIVE`
(two) play at once: one thing you notice and one thing you feel; past that the frame
stops being a photograph. The shape that works, and the one the craft pass produces
without any model: something **arrives** at the cut (`rack_focus`, `glitch`), a **bed**
runs under most of the hold (`breeze`, `weave`, `bloom`, `bokeh`), and on a long hold
something **leaves** near the end (`light_leak`, `pulse`). The renderer cross-fades each
beat's weight over half a second and composes the two live ones in one pass — A's warp
feeds B's, and B's colour pass sees A's output, so "a focus pull settling into a breeze"
is one continuous thing rather than two filters fighting.

**The effect library is data, and its descriptions ARE the interface** (`feed/effects.py`).
**Thirty-six shot effects and fourteen transitions**, in six families:

| family | effects |
|---|---|
| the plain push | `plain` |
| optics — what the lens does with light | `rack_focus`, `bloom`, `bokeh`, `light_leak`, `anamorphic_flare`, `god_rays`, `prism`, `soft_focus`, `spotlight` |
| colour — the same picture, developed differently | `duotone`, `bleach_bypass`, `cross_process`, `infrared`, `colour_isolate`, `flash` |
| geometry — what the camera does with the frame | `breeze`, `pixelate`, `zoom_punch`, `shake`, `dolly_swing`, `mirror`, `kaleidoscope`, `tilt`, `parallax` |
| the surface it is shown on | `weave`, `film_burn`, `vhs`, `crt`, `halftone`, `paper` |
| rhythm — things that happen ON the beat | `pulse`, `glitch`, `strobe_cut`, `freeze_ripple`, `echo` |

Transitions: `dissolve`, `cut`, `dip`, `whip`, `burn`, `slide`, `slide_up`, `zoom_blur`,
`glitch_cut`, `iris`, `page_turn`, `shatter`, `match_zoom`.

Each is one fragment-shader branch plus **four numbers packed positionally into a `vec4`**
(`PARAM_ORDER`), so no effect costs a pass, a framebuffer or a texture. A model handed
`{"bloom": "id 2"}` picks at random; a model told *what it looks like, when to use it and
when to avoid it* picks like an editor — so every entry carries `feel`, `use_when`,
`avoid_when`, and a note per parameter explaining both ends of its range.

**NOTHING MAY APPEAR FROM NOWHERE, AND NOTHING MAY SNAP BACK.** Two beats run at once and
hand over by weight, and a beat pinned to the shot's own edges is at weight 1 on its first
frame — so an effect that is full strength at t = 0 is a jump, and one still displaced at
t = dur is a snap. Every effect added beyond the first nine therefore carries its own
envelope (`beatEnv` in the shader): exactly zero at t = 0, exactly zero once the shot's
clock has run out, and zero through the overrun the outgoing layer spends inside a
transition. Geometry effects satisfy it structurally as well — a fold reflects back into
the frame, a curl pins its far edge, a parallax holds its border — so no move can uncover
emptiness. Transitions carry the same guarantee at their own scale: every mid-transition
flourish is multiplied by `sin(pi * uMix)`, which is zero at both ends, so a mode is
exactly the outgoing picture at 0 and exactly the incoming one at 1. That is the `dip` bug
of §11.1 made impossible by construction rather than by remembering it.

**The conductor is shown only what this memory can use.** The whole library is ~13 000
tokens and the brief already carries a line per photograph, so each entry declares the
moods it belongs to and `catalogue_for_prompt(energy)` sends the shortlist — 28 effects
for a lively memory, 15 for a sad one. A birthday is never offered `soft_focus`; a
hospital visit is never offered `kaleidoscope`. `tests/test_feed_effects.py` fails the
build if the catalogue ever advertises an effect or a transition the shader does not
implement, and if the no-model pools in `feed/director.py` name one that does not exist or
does not suit the energy they are filed under.

**The waiting screen is part of the film.** Preparing takes minutes, nearly all of it one
MusicGen call, so the memory's own photographs drift behind the progress bar as scattered
rotated prints — **and the viewer's own notes tumble in among them**, as bubbles at
varied sizes and angles. This run is reading those notes (§11.3), and a waiting screen
showing only pictures gives no sign of it; seeing your own words in the pile is how you
know they are part of what is being made. Positions, angles and scales are derived
server-side from each item's index, so the layout is stable across the bar's one-second
polling instead of flickering. Each stage is named
for **what it is making** — "recording the voice", "composing the music", "matching
effects to the music" — because that is the only thing the user can see during the run.

**The renderer is WebGL2, not CSS** (`web/static/feed-gl.js`). CSS can cross-fade two
`<img>` layers and nothing more; the film needs true dissolves and dips, **zoom and
directional blur** on fast moves, a per-memory **colour grade**, vignette and grain —
all of which are one fragment shader and none of which the DOM can do. The camera move
is a UV transform, so it costs no layout and holds 60 fps. There is **no ffmpeg and no
server-side render** anywhere in this pipeline — a video FILE is produced, but by
recording this same canvas in the browser (§11.2).

**Autoplay is a click.** Browsers refuse to start audio without a user gesture, so the
film opens paused behind a title card; the first click starts picture and sound.

**Language.** `feed_language` (§6, `app_settings`) is `en` by default and may be `ru`.
It selects the language the story pass writes in, and therefore the language the voice
reads. A **presentation is per language** — `memory_presentations` is keyed
`(group_id, language)` — so both can exist side by side and switching does not discard
the other. Narration needs a voice that speaks the language: Kokoro covers English but
**not Russian**, so the catalog declares each voice's `languages` and `POST /speak`
answers **503** when no offered voice can speak the requested one (§4.1). The film's
language also picks the voice MODEL automatically — the user never selects it — and a
Russian film that cannot be narrated is a failed film, not a mute one.

**Film pipeline.**

```mermaid
flowchart TB
    trigger["Prepare film (manual, background thread)<br/>one at a time per process · per LANGUAGE"] --> sig
    sig["1 · Signals (PURE SQL, no model)<br/>orientation · tags (vibe/palette/light/quality/subject)<br/>place_city · shot_at · SigLIP vector"] --> dir
    dir["2 · Direct (PURE, no model)<br/>camera move per shot (one eased PUSH, never a pan)<br/>burst → hard cuts · hero → long push-in<br/>palette+light → grade · a DEFAULT effect per shot<br/>this is the film if no model ever runs"] --> story
    story["3 · Story (model call 1, POST /text/complete)<br/>narration script · sparse title cards · music prompt<br/>+ SCORE PLAN: where the music changes, and to what<br/>written DIRECTLY in en or ru"] --> narr
    narr["4a · Narration (POST /speak, BATCH) — MANDATORY<br/>kokoro (en) · silero v4_ru (ru, split at sentences)<br/>returns segments [{text,start,end}] — the film's SPINE<br/>no script in the language, or no speech → FAIL"] --> music
    narr -- "cannot speak" --> failed["status='failed' + the reason<br/>the page shows the error, never a mute film"]
    music["4b · SCORE (POST /music, BATCH) — feed/score.py plans it<br/>2-4 sections on the story's own turns, each a CONTINUATION of the last<br/>brief per section = mood + THAT stretch's photographs + role (open/lift/turn/close)<br/>covers the runtime and ENDS — no loop · 503 when the slot is off"] --> beats
    beats["4c · Beats (PURE numpy, feed/beats.py)<br/>spectral flux → onset envelope → tempo → downbeats<br/>NO model"] --> cond
    cond["5 · Conduct (model call 2, feed/conductor.py)<br/>briefed with: every photo's description + 10 tag dimensions +<br/>place/time/EXIF · every narration sentence's timing ·<br/>every downbeat · the whole effect library<br/>returns: WHICH photo plays WHEN, with which effect + params<br/>MAY REORDER — the cat under the sentence about the cat<br/>repaired by director.with_arrangement · fallback = even + snapped"] --> persist
    persist["6 · Persist<br/>WAVs → data_dir/audio (content-addressed, §6.1)<br/>row → memory_presentations(group_id, language)"] --> play
    play["/feed/{key}?lang=… — WebGL2 player<br/>CLOCK = the soundtrack's currentTime, never a timer<br/>36 shot effects + 14 transitions, one fragment shader<br/>NO model · NO ffmpeg · a LEAF under the memory grid (§13.1)"] --> save
    save["↓ Download (§11.2) — MediaRecorder over THIS canvas + THIS audio bus<br/>real time · mp4 where the browser can encode H.264, else webm<br/>no second renderer, so the file is what you watched"]
```

**A model CASTS the film; the code only checks the cast is real.** Every speaker a
voice model offers is declared in `models/catalog.py` beside it — name, gender,
register, and a sentence saying what it sounds like (CLAUDE.md § "A model's own metadata
is DECLARED"). The story pass is shown that table together with the memory and the
viewer's notes, and returns a **`narrator`** for the film plus, per passage, an optional
different **`voice`**. Judging which voice suits which moment is taste, and taste is
what the story pass owns; enumerating what exists is data, and `feed/voice.py` holds no
list of names at all.

Two things the model is not trusted with. `voice.resolve()` checks every cast against
the catalog, because an invented name raises inside the TTS child and narration is
mandatory — a bad cast would fail the whole film. And `voice.pick_speaker()` is the
fallback for when no model runs, deriving a narrator from the memory's own mood and
photographs.

**A `[voice]` bracket repaints EXACTLY ONE PASSAGE, never the film.** The narration is
spoken in parts — `POST /speak` takes the exact chunks *and* a speaker per chunk — and
the clips are merged into one WAV, so the film keeps its narrator and returns to it after
the moment that asked to sound different. *One* passage is load-bearing: keying the
override to a **photograph** repainted every passage that showed it, and a story that
returns to one picture five times got five hoarse passages out of a single note. The
note's position through the film gives a target index; where a previous cut says which
photograph was on screen, the **nearest** passage showing it wins instead, so the
instruction follows that picture into a re-cut without spreading across every reuse.

**Same-voice passages are spoken in ONE call** (`audio/segments.py`). Synthesising each
passage separately restarts the prosody every time, so an eight-passage film came out
sounding like eight different readers even where the speaker never changed. Passages are
grouped into runs that share a voice, each run is one call, and the per-passage timings
are recovered by splitting the run's audio in proportion to its text. That split is an
approximation, bounded inside one continuous utterance — a far better trade than exact
boundaries around a delivery that lurches every few seconds.

**One memory can be retold on its own**, from a **↻ Rebuild** on its Organize card —
on the memory, not on the film screen, because it changes the memory and every film made
of it follows (`POST /feed/{key}/rebuild`,
`memories_build.rebuild_memory`). Rebuilding *every* memory re-seeds the clusters, so a
memory you liked can come back split or merged — a heavy, whole-library operation for
what is usually one dissatisfaction. The per-memory button asks the same agent one
question about one memory, with that memory's **current photographs** as the candidate,
so it is re-ordering and re-describing what is already there rather than re-clustering
anything.

It rewrites the memory **in place** (`memory_store.rewrite_memory`), and that is
load-bearing: `replace_memories` drops every group row and takes the group ids with it,
and `memory_notes` cascades on those. The viewer's own words must survive a retelling —
they are the input to it. What does *not* survive is the film: every stored presentation
for that memory is dropped, because it narrated a memory that no longer exists in that
shape. A retelling that fails writes nothing and the old memory stands.

**THE PHOTO ORDER BELONGS TO THE MEMORY.** It is decided once, by the memory-composing
agent (`albums/compose.py`), which now returns an `order` alongside `keep`/`title`/
`description`: every kept photograph exactly once, pictures of the same thing placed
next to each other so they read as one beat, and those groups sequenced the way the day
would be told. The **description is written for that order**. The Organize album shows
it, and **every film of that memory follows it** — `director.with_script` re-sorts the
story pass's passages so a photograph can never be introduced out of sequence.

The story pass still chooses which photograph carries each passage, how long the film
dwells on one, and whether it returns to it: **a repeat is not a reorder**, so the
constraint is on first appearances only, applied as a stable sort that leaves repeats
where the story put them. The order changes only when the memory itself is rebuilt.
Without this the same memory came out in a different sequence on every regeneration and
never looked like itself twice. The agent's `order` is repaired, never trusted: invented
ids are dropped and forgotten ones appended, so a memory cannot silently lose a
photograph it decided to keep.

**THE PROMPT IS CUT TO THE SERVER'S CONTEXT.** `llama-server` does not truncate a
prompt that does not fit — it rejects the whole call ("request (6306 tokens) exceeds
the available context size (4096 tokens)"), the story comes back empty and the film
fails for want of narration, naming the language rather than the real cause. So the
story pass asks the models service what the running server accepts
(`GET /text/context`, §5.1) and budgets against it: everything else in the prompt is
assembled first, and the **photo listing** — the only part that grows without limit,
one ~130-token brief per photograph — is given what is left after the answer is
reserved, shared equally and clipped per line. **Numbering is the contract** (the
story answers with `shot` numbers), so a brief is clipped, never dropped. Below a
floor of one short line per photograph nothing more can be cut; that memory is over
the context however it is written, and it is logged as such rather than failing
silently. Token counts are estimated from characters (~3.5 measured; 3.0 used, so the
estimate errs towards trimming): `app` is a thin client and holds no tokenizer.

**A NOTE THE MODEL WILL NOT WRITE IN IS PUT IN ANYWAY.** Asking twice is not a
guarantee, and it was measured not to be one: on a real film gemma4-E2B dropped the same
note through three different attempts — a word budget raised to cover it (it then wrote
45 words of the 115 it was given), an explicit complaint on retry, and a required schema
field forcing one entry per note (it filed the FIRST note twice). A 2B model composing a
whole story around a fact is the hard version of the task. So after the retry, anything
still uncovered gets the easy version:

1. **One small call rewrites ONE passage** to carry that one fact, in the film's own
   language — the same reason the language fallback is a translation pass rather than
   another attempt. The passage is the one at the note's own share of the film.
2. **If that does not carry it either, the viewer's own words go in verbatim**, as their
   own passage, at their position, over the photograph they were written about. A
   sentence that reads a little flat is worth more than one that is missing.

A `[bracket]` is stripped before either — it is an instruction, never narration (§11.3)
— and a note written in a language the film does not speak is left to the model rather
than fed to a voice that would read it phonetically.

**The word budget PAYS FOR THE NOTES.** `narration_words` adds 30 words per note on top
of the photographs' share. It used to count photographs only, so a three-photograph
memory was told to write 55-75 words in total while carrying two notes worth 32 words of
mandatory content — the model obeyed the cap and dropped one, every time.

**The story pass CHECKS ITS OWN OUTPUT, and asks again.** Telling a model that every
note matters is not the same as it obeying, and each way it disobeys is silent — the film
looks fine and something the person said is gone or in the wrong place. So three things
are verified after the call, and any that fail are named back to it in one retry:

- **coverage** — a note whose content appears nowhere (`uncovered_notes`);
- **placement** — a note whose words landed far from where it was written
  (`misplaced_notes`), measured in shares of the film rather than seconds because the
  new cut is a different length. This exists because the coverage retry *caused* it:
  told only to include a missing note, the model appended it to the last passage, so a
  remark about the middle of the evening was read over the closing shot. The complaint
  names both positions — "you put this about 92% in; it belongs about 49% in";
- **casting** — a note carries a bracket and no passage was given a voice.

The retry is kept only if it answered MORE than the first attempt, so a worse second try
cannot make the film worse. None of these read what a bracket MEANS: they notice that
something was asked and not answered, which is a different thing from deciding the
answer.

A share is quoted only when a previous cut says how long the film was. Falling back to
the notes' own span made the last note "100% of the way through" every time — it is the
one that defines the span, so that is an artefact, not a position.

**Notes are a SEQUENCE, not a bag of facts.** They are listed to the story pass in time
order, and the prompt states that this order *is* the order of the day: a note at 0:07
describes something that happened before one at 0:11, and the passages must follow the
same sequence. A story that tells the fourth and fifth and then doubles back to the third
contradicts itself.

**THE SCORE IS A SCORE, NOT A CLIP** (`feed/score.py`). The film's music is two to four
**sections** that follow the story's own turns, generated in one call as one continuous
piece: section two is a **continuation** of section one's last two seconds, so it
arrives in the same key and tempo even when its brief asks for something different, and
the join is crossfaded over the model's own reconstruction of what it was given. The
score covers the runtime and **ends** — the player no longer loops anything, because a
loop is both a seam in the middle of the story and the reason every film sounded like
one bed under different pictures.

Three things write each section's brief, and the order is the order of authority:

- **the memory's measured mood and occasion** (`feed/mood.py`), first, for the reason
  above;
- **the photographs THAT SECTION is under** — its own stretch of `season_weather`,
  `light`, `setting` and place. This is what a single prompt cannot do at all: it can
  only ever describe the average of the whole memory, so snow in the last third either
  colours the entire film or none of it;
- **the role** — `open` is sparse, `lift` adds percussion, `turn` changes the lead
  instrument, `close` resolves and fades. Without it the sections are four variations
  on one idea rather than a piece that goes somewhere.

**The story pass says WHERE the music changes.** It wrote the passages, so it knows
where the film turns; its `score` plan is `[{from: passage, brief}]` and it decides the
boundaries, while `feed/score.py` still writes the measured half of every brief. With no
plan the film is split evenly, every boundary dragged to the nearest **passage start**
(never mid-sentence, `SNAP_WINDOW`), and the story's one film-wide sentence colours each
section. Nothing here can fail a film: music has always been the one artifact a memory
may go without.

**It costs what it covers.** MusicGen is autoregressive and measured at ~4.9x realtime
for `musicgen-small` on an M-series mac (15 s of audio in 73 s), so a 90-second film is
about seven minutes of generation and `SCORE_MAX_SECONDS` (180 s) is the ceiling that
keeps a long memory from doubling a wait nobody asked for. The seed is derived from the
memory, its language and its signature, so the same film scores the same way twice and
no two memories share music.

**The MOOD decides the score, the voice and the pace** (`feed/mood.py`). One vote across
every photograph's `vibe`/`emotion`/`occasion` gives `high`, `calm` or `low`, and three
things obey it so the film cannot contradict itself:

- **the score** — the mood clause is put FIRST in every section brief and names a tempo
  in BPM and real instruments. A small model reaches for "warm ambient pads, gentle"
  whatever it is shown, and that is funeral music over a birthday; the story pass's own
  sentence is kept after ours as colour, not as the brief. The energy also sets HOW the
  score is sampled (`mood.SAMPLING`): a party is generated with more guidance and more
  heat than a quiet afternoon, where MusicGen's one default setting for everything was
  itself a reason every film came back sounding the same;
- **the narrator** — the energy picks the band and the memory's own photo ids pick which
  voice within it, so two quiet afternoons do not sound like the same person reading
  twice. Every name is verified to load and produce audio before being offered, because
  narration is mandatory and an unknown speaker FAILS a film rather than degrading it;
- **the pace** — how many beats a shot cycles through and how fast they hand over
  (`mood.PACE`), and which effects are reachable at all: `glitch` and `pulse` belong to
  a party, not to a hospital visit.

### 11.2 Taking the film with you

A film you cannot keep is a demo. **`↓ Download` records the film to a video file**, and
it records **this canvas and this audio bus** — the same WebGL context the player just
drew and the same WebAudio mix it just played through. There is no second renderer, so
the file cannot drift from what was on screen, and the effect library grew to thirty-six
shaders and fourteen transitions without an export path to keep in step behind it.

Consequences of that choice, both accepted deliberately:

- **It records in REAL TIME.** `MediaRecorder` muxes a live audio track and there is no
  way to hand it a faster-than-realtime one without shipping a muxer into the page, so a
  50-second film takes 50 seconds to write. The button says so, shows a percentage, and
  is its own Stop.
- **The container follows the browser.** MP4 (H.264 + AAC) where the browser can encode
  it — Chrome ≥ 130, Safari — and WebM (VP9 + Opus) otherwise. A playable WebM beats an
  MP4 the browser refuses to produce. A browser with no `MediaRecorder` at all is told
  so on the button rather than handed a broken file.
- **NOTHING IS RECORDED UNTIL THE FILM CAN PLAY WITHOUT STOPPING.** Because it records
  in real time, whatever the film does while the recorder runs is in the file — and on a
  cold page neither WAV is buffered (`preload` is a hint, not a promise) and no
  photograph is in the browser's cache, since textures are fetched as each shot comes
  up. Starting the recorder and playing in the same tick therefore RECORDED THE
  BUFFERING: the first download stalled and the second, after a reload had warmed the
  cache, was smooth. So the export now waits for `canplaythrough` on both tracks and for
  every photograph to decode before it starts, says "Preparing…" while it does, and
  treats a click during that wait as "never mind". Both waits are bounded (20 s), because
  a browser that never fires `canplaythrough` must not hang the button.

**There is still no ffmpeg and no server-side render.** Exporting on the server would
mean reimplementing every GLSL effect as a filter graph and maintaining two renderers
that must agree frame for frame — the exact drift this app avoids everywhere else. The
canvas already is the renderer; recording it is the whole feature.

**One consequence for the audio graph:** `createMediaElementSource` may be called only
once per `<audio>` element, so the mix is built once into a single `bus` node, and the
recorder taps that bus. Building a second graph for the export throws and silences the
film.

### 11.3 The viewer's own words

Generation knows what the machine could work out. It cannot know what the day MEANT to
the person who lived it — who that is in the photograph, why that evening mattered, what
the caption model got wrong. So **the film can be annotated while it plays**.

Pausing opens a panel: every note already on this film, editable, plus a new one stamped
at wherever the film is standing. Notes may be in **any language**. They are stored in
`memory_notes(group_id, at_seconds, text)` and cascade with the memory, like a
presentation.

**The editor is hidden only while the film PLAYS.** Watching is the one moment a form
over the picture is wrong; every other state wants it. So it is open on the title card
(where you read what you already said before deciding to watch or to prepare again), on
a pause, and on the **Prepare** screen where there is no player at all. It is hidden
outright while a film is being generated, along with every other control except close
and cancel. Bubbles frozen mid-flight stay **behind** it: context, never
over the thing being typed into.

**The panel is a sheet that folds.** It overlays the memory's own description on the
Prepare screen, and covering the thing you are writing notes *about* is not a trade
worth making — so the whole body slides down into its title bar, which keeps showing the
count. The choice is remembered per browser, because reopening a film and finding the
opposite state is the panel arguing with you. Each field is **sized to its text** rather
than scrolling inside itself; the sheet takes the scrollbar when there are more notes
than fit.

**The panel does not belong to the player** (`web/static/feed-notes.js`). It renders on
the **Prepare** screen too — the one where the film does not exist yet, which is exactly
where a note is worth writing, because it is then the only thing that can steer the
first cut. That screen has no canvas, no clock and no `feed-gl.js`, so the panel is its
own script: the player borrows it through `window.feedNotes` to float bubbles and to
stamp a note with the moment it paused at, and everything else works without one.

Two things then happen with them:

- **They are visible wherever the memory is, behind a fold.** On its Organize card, and
  under the memory's own title when a photo is opened inside it — above the
  photograph's words, because a note describes the memory rather than that picture.
  Living only inside the player would mean a note can be found only by playing the film
  again. Both are a collapsed `<details>` labelled "N notes for the film": they steer
  the next generation, so they matter while you are working on the video and are noise
  the rest of the time. Each timestamp deep-links back into the film at that second.
- **They float during playback, one at its own second, and only while the film is
  actually running.** Each rises up a right-hand column — the side the panel is on, so a
  note appears where it was written and never over the middle of the photograph — and
  drifts out of the top. Emission is gated on `playing` and each note fires once; the
  record of what has been shown is cleared on a **seek or a replay**, as events, never
  by testing the clock. (A clock test is true on every frame the film sits near it: on
  the title card, where the clock rests at 0 for ever, a 0:00 note was emitted sixty
  times a second until the screen was a wall of them.) They
  live in the **DOM, never on the canvas** — the export records
  `canvas.captureStream()`, so notes are absent from a downloaded film by construction.
  They are working notes for the person making the memory, not for whoever receives it.
- **They rewrite the next cut, and they are FACTS, not splices.** The story pass reads
  them **first, above every machine label** — a label is a guess, this is testimony from
  someone who was actually there — and each one is stamped with three things: the
  timestamp, its position as a **percentage** of the way through (a model writing a list
  of passages knows what "40% in" means and has no idea how long 0:12 is), and **which
  photograph was on screen** when it was written, resolved from the previous cut.

  That last one is what makes a note a fact rather than a local patch. "This is Julia"
  written over photograph 3 means the woman in photograph 3 *is* Julia — named in the
  passage about that photograph, named again wherever it comes back, named in any other
  photograph that is plainly her, and named in passages **before** that point too,
  because the writer knows who she is while writing them. Rewriting one sentence around
  the timestamp and leaving her "a woman" everywhere else is the failure the instruction
  exists to prevent. Where a note contradicts a caption, the note wins and the caption's
  word is dropped from the whole film.

- **Regenerate says so.** A note changes nothing until the film is rebuilt, and nothing
  else on the page admits that. So writing one lights the Regenerate control and keeps
  it lit — even while the film plays, where the controls normally fade back. A note
  written and then forgotten is worse than one never written: the viewer believes they
  have been heard.

Editing a note to nothing deletes it: this is a scratchpad, not a form, and notes save
on blur rather than behind a button. A note lost because it was not "submitted" is the
whole feature failing quietly.

**`[Brackets]` are instructions, not sentences** (`feed/directives.py`). Sometimes the
viewer does not want to describe the moment, they want to DIRECT it. So anything in
square brackets is lifted out of the note and routed to whichever stage can act on it,
and the rest stays as prose — `[low man's voice] it was a bad day` gives the voice a
register *and* the story a sentence, and the story pass never sees the bracket it would
otherwise try to narrate.

**NOTHING PARSES A BRACKET.** There was a matcher — a table of phrases, then of word
stems — that classified each bracket and routed it. It is gone, and it was wrong in a
way no amount of table-filling fixes: `[страшных хриплый мужкской голос]` fell straight
through it, because it knew `хриплый голос` and another word sat between the two. People
write instructions in their own words, in their own order, in either language, with
typos; a table only ever recognises the phrasings someone thought to add.

So notes go to the models **verbatim**, and each pass is told that brackets are
instructions to obey rather than sentences to narrate. The story pass reads them and
casts the narrator, any per-passage voice, and the score; the conductor reads them and
places effects and lengths. Interpreting "a scary hoarse man's voice" into a speaker is
judgement, and judgement is what a model is for. Code keeps only the parts a model must
not be trusted with: `voice.resolve()` checks a cast is a real speaker, `effects` checks
a named effect exists, and the deterministic fallbacks decide nothing more than what the
film looks like when no model runs.

The categories below are what the on-screen legend shows as EXAMPLES of what can be
asked for — not a vocabulary, since a model is reading it:

| category | what acts on it | example phrasings |
|---|---|---|
| `voice` | **that PASSAGE alone is spoken by another voice.** The film keeps its own narrator; the narration is synthesised per passage and the clips merged | `[low man's voice]` `[страшным хриплым голосом]` `[whisper]` `[excited]` |
| `music` | **leads the score brief**, ahead of the measured mood and ahead of the story pass's own sentence | `[fun chord]` `[swell]` `[drums]` |
| `effect` | **`feed/conductor.py`**, on the shot covering that timestamp, told not to substitute a nicer-looking one | `[glitch]` `[bloom]` `[slide]` `[light leak]` |
| `timing` | **`feed/conductor.py`**: a named length sets that shot's `seconds` and its neighbours absorb the difference; `skip` drops the photograph, `title` gives it a card | `[hold 5s]` `[quick]` `[skip]` `[title]` |
| anything else | passed to the story pass verbatim | `[like a fairytale]` |

Every prompt that receives notes says in as many words that bracketed text is an
**instruction, not inference, and outranks its own judgement**.

There is deliberately no `sound effects` category: this app generates a score and a
voice and nothing else, and a bracket the pipeline cannot honour would be a lie.

**Either language, whatever the film is in.** A note may be written in English or
Russian regardless of the language being generated. The *prose* half is translated by
the story pass into the film's language; the *bracket* half is matched against a table
of Russian aliases (`[низкий мужской голос]`, `[весёлый аккорд]`, `[держать 5с]`),
because a direction is a switch rather than a sentence and matching it is cheaper and
surer than asking a model to translate it. The legend shows both forms.

A trailing number is kept, so `[hold 5s]` differs from `[hold]` — and `с`/`сек` count
as seconds too. Every effect and
transition name in the library is automatically a valid bracket, and an **unrecognised**
bracket is not an error — it is passed on verbatim, because a person writing `[like a
fairytale]` means something even though we never enumerated it. The on-screen legend is
generated from the parser's own vocabulary, so it can never advertise a word the
pipeline ignores.

Directions **outrank the pipeline's own judgement**: they are the one thing in the whole
prompt that is not a guess.

## 12. Stage 2 — reorganizing your folders on disk

Stage 1 learns what the photos are. Stage 2 acts on it, on the machine that holds
them. The whole contract between the two is **one JSON plan document**, downloaded
from Organize — and the local `ivms777-sync` CLI is the only component that ever
writes to your disk (§3.2b). **The plan schema, the CLI commands, and the full
safety rules live in [`docs/sync-cli.md`](sync-cli.md).** The design:

- **The organizer IS the layout.** There is no separate layout picker, no `Layout`
  protocol, and no `/export` screen. The principle already chosen in Organize
  (§11) — date at day/month/year, camera, place, memories — is exactly the folder
  structure the plan produces, so adding an organizer adds an export for free.
- **The plan** (`GET /api/plan?by=…&grain=…`) runs that organizer and lists, per
  photo, its content hash, the `target` relative path its album assigns, and every
  source path it was uploaded from; plus, per album, the folder name, title, and
  description. It is derived state: a different organizer re-derives every `target`
  and nothing else. It carries a `complete` flag (§8).
- **Reorganization happens in place, inside a source folder.** The tree is built
  under the root the CLI is pointed at; files never move between folders and their
  **filenames never change**. One plan covers the whole library — run it in each
  source folder you want reorganized, and hash matching means each run touches only
  the files actually present there.
- **Every album folder gets a `description.txt`** — title, description, photo count.
  For a memory that is the agent's full description, so the story survives outside
  the app.
- **A photo can be in several memories; a file cannot be in several folders.** The
  export resolves it: the **event** memory owns the file, and a **theme** memory
  becomes a folder holding `description.txt` plus a `photos.txt` listing where its
  photos actually live (§11). Photos no album claims go to `_unsorted/`.
- **The CLI is `apply` / `undo` / `verify`.** It matches files by content hash,
  never by path, so a library reorganized since upload still matches. `apply` needs
  no network — the plan document is self-contained.
- **Safety is paranoid:** nothing is ever deleted (redundant copies move to
  `_duplicates/`), nothing outside the plan is touched, every operation is
  journaled before it runs and reversible by `undo`, cross-filesystem moves are
  copy-verify-unlink, and `apply` refuses an incomplete plan unless told otherwise.

## 13. UI

The app is a **persistent HTMX-boosted shell** — a fixed full-width nav with a live
resource bar (RAM/CPU always, GPU and board temperatures where readable) over a
single scrolling `<main>`
that the four top-level links swap without re-rendering the nav. The routes:

- **`/upload`** — the folder list (add / delete-from-library, §3.2c), then
  client-side hashing, transfer, and live per-stage processing progress with
  per-stage **Reprocess** buttons (§8).
- **`/library`** — infinite-scroll grid with a faceted sidebar (model tags + EXIF
  facets, apply-on-change), sort, search box, and parsed-filter chips (§9).
- **`/photo/{id}`** — the one **leaf** view: the photo shown inside the collection it
  was opened from (`ctx`), its AI title/description, tags, full EXIF (incl. the GPS
  coordinates, which live only here), source paths, and an async "similar" strip with
  why-it-matched reasons (§9). All drill-down navigation obeys §13.1.
- **`/organize`** — a principle dropdown (date grain / camera / place / memories) over
  album cards; memories has the background Rebuild control; the last-opened view is
  remembered. An **Organize files…** button turns the current principle into a
  stage-2 plan: `/organize/plan` previews the folder tree it would create, asks for
  confirmation, and downloads `ivms777-plan.json` (§12).
- **`/feed/{key}`** — the fullscreen **memory film**: one memory played as a directed
  short with camera moves, cuts, a colour grade, a generated soundtrack and optional
  narration (§11.1). A language selector (English / Русский) sits on the page; each
  language is prepared and cached separately.
  Reached from a ▶ Play control on a memory card in `/organize?by=memories`. A
  memory with no prepared presentation is offered one, and the page polls while
  the background job runs; otherwise it plays. It is a **leaf** under the memory
  grid (§13.1), and it is the one page that does not use the app shell — a
  fullscreen player has no room for the nav, and the nav must not boost it.
- **`/chat`** — the persisted conversation with streamed answers and thumbnail
  citations (§10).
- **Settings (⚙, in the nav beside the resource bar)** — a modal `<dialog>` served
  by `/settings/models`: one section per model slot (§4.1), the selectable catalog
  entries as radios with the current one checked, each showing its download size,
  RAM cost (flagged when estimated), and either "on disk" or a live download bar.
  An entry that is **part** on disk says so and offers only the remainder — two
  slots share a file whenever the caption entry is the planner's weights plus a
  vision projector, so the same model reads "on disk" in one slot and "absent" in
  the other, and offering the full size there would overstate it by GB.
  Choosing a different model states what the switch re-runs and for how many
  photos, and only then offers **Switch**. It is an **overlay, not a layer** — it
  pushes no history entry and changes no URL, so §13.1's `[grid, leaf]` invariant is
  untouched, and Esc/close simply closes it.

**Per-route detail — the resource-bar internals, the full `/photo` panel, the feed
player's controls, throughput readouts — is in [`docs/ui.md`](ui.md).** The nav order **Upload → Library → Chat →
Organize** follows the workflow: bring photos in, browse, understand, then (last)
reorganize — Organize feeds stage 2.

### 13.1 Navigation model (STRICT — every drill-down obeys this)

The UI is **grids** and **leaves**. A *grid* is a browsable list — the library
(with its filters/search/sort), an Organize album, a memory. A *leaf* is a detail
view (`/photo/{id}`). The rules below are absolute; every current and future
drill-down follows them exactly.

**The persistent shell does not touch this model.** Only the four top-level nav
links are HTMX-boosted (§13); every grid↔leaf move here is a plain `<a href>` full
navigation, so the `history.back()` + bfcache behaviour rules 3–4 rely on is
unchanged. Never boost a drill-down link — doing so would put htmx's history cache
in front of bfcache and break the `[grid, leaf]` invariant.

1. **A leaf is always exactly ONE level below a grid.** There is no photo-inside-a-
   photo nesting. A "similar" photo is still just a leaf one level under a grid.
   **The memory feed (`/feed/{key}`, §11.1) is a leaf too**: one level under the
   memory grid, it never opens a photo leaf from inside itself (that would be
   leaf-inside-leaf), and every shot change is player state — never a history entry.

2. **The leaf records its grid in a `ctx` URL parameter** — never guessed:
   - `ctx=library` (plus `q`/`f_`/`n_`/`t_`/`sort`/`date_*`) → the library grid.
   - `ctx=album:<by>:<grain>:<key>` → an Organize album grid.
   - `ctx=similar:<id>` → the library grid, with the origin photo `<id>` as an
     intermediate **step on the way up**: it is shown as clickable context (a
     thumbnail in the header) AND it is what close returns to first. Close from
     there goes up to the library. It is the one ctx whose close takes two steps.
   - The **feed** (`/feed/{key}`) needs no `ctx` at all: the key IS the memory's
     album key, so its grid is never guessed. An optional `ctx` supplies only the
     deep-link `href` fallback that close uses when the feed was opened cold from a
     pasted URL.
   - `ctx=chat` → the grid is the **conversation**; close returns to `/chat`. A
     cited thumbnail carries this ctx, and it pages within the photos the current
     chat session cited (its evidence set), never the wider library.
   - `ctx=chat-memory:<key>` → a **memory shown in chat**. The grid is that memory
     (resolved by the Organize album `<key>`), so it pages within the memory's
     photos and shows the whole-memory collage exactly like the Organize leaf — but
     close returns UP to `/chat`, the conversation it was surfaced in, not to
     `/organize`. It is the one ctx whose grid (a memory) and close-target (the
     chat) differ, because the memory is browsed *through* the conversation.

3. **History invariant: `[grid, leaf]` — depth two, always.** Grid→leaf (clicking
   a photo in a grid) is the ONE `push`. Every move at the leaf level —
   **prev/next paging, opening a similar photo, clicking the origin thumbnail, and
   closing a similar photo back to its origin** — uses `location.replace`, never a
   push. History therefore never accumulates a chain of visited photos, and the
   two-step close of rule 4 costs no extra depth.

4. **Close / Esc goes UP exactly ONE level.** For a normal leaf that level is the
   grid: `history.back()` (so the grid's scroll and filter state restore via
   bfcache), with the computed grid URL (`origin_url`) as the deep-link fallback.
   Because history is `[grid, leaf]`, back always lands on the grid — it can never
   "replay" photos, because none are in history.

   **A `similar:<id>` leaf has one extra level above it — the origin photo.** Close
   there goes to `/photo/<id>` **in place** (`location.replace`, so the history entry
   is swapped, never pushed), and closing *that* runs the normal `history.back()` to
   the grid. Two closes: similar → origin → grid. The replace is what keeps rule 3
   intact — the depth stays two, the leaf slot simply now holds the origin photo.

   **A LEAF MAY HOLD LEVELS OF ITS OWN, and Esc unwinds those first.** The film (§11.1)
   is the case: a recording in progress and a playback in progress are each a state the
   viewer is inside, and closing the page out from under one is not "up one level", it
   is losing their place — the rule this whole section exists to protect. So Esc there
   is: cancel the recording → stop the playback and return to the title card → and only
   with nothing left to undo, `history.back()` to the grid. The film answers whether it
   consumed the key (`window.feedFilm.escape()`); when it does not, the page closes
   exactly as before.

5. **Prev/next move only *within* the current layer's order** (owner-scoped),
   carrying `ctx` forward — never leaking into a sibling album/memory or the wider
   library.

6. **The grid keeps the user's place.** Filters, search, sort, and scroll survive
   the round trip (the "never lose the user's place" rule).

The recurring bug this prevents: pushing on a photo→photo move makes
`history.back()` walk backwards through every photo visited instead of returning
to the grid. If a drill-down ever "replays" photos on close, it pushed where it
must replace.

## 14. Code layout

The source uses a **flat layout**: each top-level package (`config`, `db`,
`storage`, `inference`, `embedding`, `ingest`, `organize`, `search`, `albums`,
`feed`, `audio`, `chat`, `web`) sits at the repo root — no wrapping package
directory. **The full annotated tree is in
[`docs/code-layout.md`](code-layout.md).** Three boundaries the architecture leans
on:

- **One OpenAI-compatible HTTP client** (`inference/client.py`) covers every
  profile — `llama-server` (mac/jetson) and vLLM (cloud) differ only in `base_url`
  and model name.
- **`feed` and `audio` are opposite sides of the one-model-process line** (§5.1).
  `feed` builds and stores memory presentations (§11.1) and is a thin client — it
  holds no model and is covered by the import gate. `audio` is models-service-only:
  it hosts the TTS and music models inside `TorchWorker` children, so — like
  `modelsvc` — it is exempt from that gate and is never imported by `app`, `worker`
  or `feed`. `storage/wav.py` is the one piece both use, and it is torch-free
  precisely so it can be.
- **`ivms777_sync` is a standalone package** that imports nothing from `ivms777` —
  no database, no Pillow, no models, only the stdlib plus `httpx` — so it installs
  on a user's machine in seconds and runs anywhere Python does.

## 15. Testing

The testing strategy — deterministic fakes, PIL-generated fixtures,
integration/route/upload tests, `owner_id` scoping tests, and `ivms777_sync`
plan/apply/undo + failure-injection tests — lives in
[`docs/testing.md`](testing.md).

## 16. Phases

The delivery phases (0–7) live in [`docs/roadmap.md`](roadmap.md). Each phase
leaves a working, useful application; phase 7 (stage-2 sync) is last because the
plan is richer the more the library knows — memories need the agent — but it
depends on nothing after phase 3 and can be pulled forward.

## 17. Risks

The risk register — single-writer contention, 8 GB residency, chat precision,
resumable indexing, HEIC/RAW handling, `sqlite-vec` portability, and
`ivms777-sync` data safety — lives in [`docs/risks.md`](risks.md).

## 18. Future work

The future-work backlog lives in [`docs/roadmap.md`](roadmap.md): auth/quotas
(v02), face clustering, object storage, XMP export, a read-only MCP server, video
support, an `ivms777-sync` watch mode, and a browser-side apply wherever the File
System Access API is actually available (§3.2b) — plus the
already-shipped retrieval-core and chat-degradation items kept there for history.
