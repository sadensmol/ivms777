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
| `voice` — memory-film narration (§11.1) | `Kokoro-82M` (en) · `Silero v4_ru` (ru) · `VoxCPM2` = saved voices (en, ru; mac/cloud). The `voxcpm2` voice child is built with LoRA layers for baked voices: about 18 MB in float32 (an estimate), inside `cost_mb` 10500; with no adapter loaded they add no adapter contribution, and their extra matmuls are not measured. | `TorchWorker` child, `kokoro` / `torch.package` / `voxcpm` — MPS (mac) / CUDA (jetson, cloud) | `voice` |
| `music` — memory-film score (§11.1) | `MusicGen medium` (mac) · `MusicGen small` (jetson/cloud) | `TorchWorker` child, `transformers` — MPS (mac) / CUDA (jetson) | `music` |

Each slot is switchable at runtime from the settings popup (design §4.1, §13);
the table gives the defaults. `caption` and `planner` share one `llama-server`
child, so their two residency units are mutually exclusive. `voice` and `music` are
the two **optional** slots: their `off` entry registers no residency unit at all, so
the governor never budgets for a model the user turned off.

One more model runs in the `models` service and is **no slot**: the speech recognition
of a take, `whisper-large-v3-turbo` (`openai/whisper-large-v3-turbo`, mac and cloud), in
the `asr` residency unit (a `TorchWorker` child, `transformers` on MPS or CUDA). It hears the
words of a saved voice's takes right after they are saved (films and tests send a take
with its words), and a bake check's bake asks for it too; nothing else uses it. It is
not in the settings popup and has no default per profile ("Hearing the words of a take
— Whisper", below).

A second model is **no slot** either: the speaker-similarity meter's
`ecapa-voxceleb` (`speechbrain/spkrec-ecapa-voxceleb`, mac only), in the `speaker_embed`
residency unit (a `TorchWorker` child, `speechbrain` on MPS). It compares a voice test
with the voice's own takes and only measures: a film never calls it ("The
speaker-similarity meter — ECAPA-TDNN", below).

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
| `voice` | `kokoro-82m` (default, English) · `silero-v4-ru` (Russian) · `voxcpm2` (saved voices — voice copy, English + Russian; mac/cloud). The `voxcpm2` voice child is built with LoRA layers for baked voices: about 18 MB in float32 (an estimate), inside `cost_mb` 10500; with no adapter loaded they add no adapter contribution, and their extra matmuls are not measured. |
| `music` | `musicgen-medium` (default on mac, mac-only) · `musicgen-small` (default on jetson/cloud) |
| *(no slot)* `catalog.TRANSCRIBERS` | `whisper-large-v3-turbo` (the speech recognition of a take — its words, heard after it is saved; English + Russian; mac and cloud). Outside `CATALOG` and `SLOTS`: no user choice; `transcriber_for(profile)` names it. |
| *(no slot)* `catalog.SPEAKER_EMBEDDERS` | `ecapa-voxceleb` (the speaker-similarity meter — a 192-value speaker embedding of a clip; mac only). Outside `CATALOG` and `SLOTS`: no user choice; `speaker_embedder_for(profile)` names it. |

Everything beyond the defaults carries an **estimated** `cost_mb` until it is
measured on the board the way §8.1 requires (the drop in
`psutil.virtual_memory().available` across a load). Until then the settings popup
labels it unverified, and `test_every_model_fits_its_profile_budget` still refuses
to offer any entry whose `cost_mb + headroom` exceeds the profile budget.

### The two feed audio models

| | Kokoro-82M | MusicGen small |
|---|---|---|
| Repo | `hexgrad/Kokoro-82M` | `facebook/musicgen-small` |
| Licence | Apache-2.0 | **CC-BY-NC 4.0** (weights) — personal use only; the code is MIT |
| Size | 82M params, ~330 MB | small 300M, ~2.2 GB · **medium 1.5B, ~7.6 GB (the mac default)** |
| Output | 24 kHz **mono**, 54 voices | 32 kHz **mono**, any length (a 2-4 section score covering the film, §11.1) |
| Speed | — | MEASURED on an M-series mac: medium 4.75x realtime, small 4.90x — the bigger model costs resident size, not time |
| Cost | ~1200 MB, **estimated** | ~2600 MB, **estimated** |
| Driver | `kokoro` | `transformers` (`MusicgenForConditionalGeneration`) |
| Languages | 9, **Russian NOT among them** | n/a |
| Gotchas | needs the `espeak-ng` system package (its G2P shells out to it), and pulls `spacy` transitively | non-commercial weights — never ship it in a commercial build |

**Why MusicGen and not Stable Audio Open.** Stable Audio Open 1.0 was the first
choice — better on paper, 44.1 kHz stereo, a permissive licence. It does not work
here, and the reason is worth writing down so nobody re-picks it:

- Stability's own `stable-audio-tools` hard-pins `torch==2.7.1` and pulls a transitive
  `numpy==1.23.5`, both unsatisfiable against this project's `torch>=2.13` /
  `numpy>=2.5` — `uv sync --extra models` fails outright with it declared.
- Driving the same checkpoint through `diffusers` resolves cleanly, but its sampler is
  `CosineDPMSolverMultistepScheduler`, which is **SDE-only** and therefore drives
  `torchsde`. On this stack `BrownianInterval._split` recurses without terminating
  near the smallest sigma: raising the recursion limit to 20 000 did not help, CPU
  failed identically at step 7 of 8, and the scheduler exposes no ODE alternative.
- It is also a **gated** repo, so an unattended machine stalls on a licence click.

MusicGen has no scheduler at all — it is an autoregressive transformer under plain
`transformers` — so there is nothing in a sampler to go wrong. The price is quality
and licence: 32 kHz mono, and CC-BY-NC weights.

Both audio models are `TorchWorker` children at `Priority.BATCH`, needed only while a
film is *prepared* and never while one plays.

A free-form "type any HF id" field is deliberately **not** offered: `cost_mb`,
`dim` and `preprocess` cannot be discovered at runtime, and each breaks something
different when wrong (OOM / unloadable slot, an unusable vector table, silently
degraded embeddings).

### Voice copy — VoxCPM2

`voxcpm2` is the one `voice` entry with `clones=True` (design §4.1): it speaks a
SAVED voice from an example clip that `POST /speak` brings in `references` (design
§5.1), and it declares no preset `speakers`. Preset routing never picks it, and a
saved-voice request never reaches Kokoro or Silero.

| | VoxCPM2 |
|---|---|
| Package | `voxcpm==2.0.3` (tag `2.0.3` = commit `19b6bf7`), in the `models` extra |
| Weights | `openbmb/VoxCPM2`, pinned: `HfSource(revision="32279effe8c19989596f05d353d1447f51d9e915")` |
| Licence | Apache-2.0 — code and weights |
| Size | 2,290,004,544 params; `model.safetensors` 4,580,080,592 B + `audiovae.pth` 376,951,122 B (`size_mb=4728`) |
| Output | `tts_model.sample_rate` — 48 kHz **mono** |
| Cost | `cost_mb=10500`, **estimated**: float32 on MPS = 2,290,004,544 × 4 B + the 377 MB AudioVAE ≈ 9.1 GB, plus ≈ 15 % runtime (design §8.1) |
| Profiles | `mac`, `cloud` (jetson lists no voice copy) |
| Languages | `en`, `ru` — the two languages the app films in |
| Driver | `audio/voxcpm_voice.py:VoxCPMVoice`, a `TorchWorker` child (`VOICE_TARGETS["voxcpm2"]`) |

**Loading.** The child gets `(repo, device, revision, lora)` — `lora` is the LoRA
JSON of the entry's `AdapterSpec` (see "Voice baking" below). It refuses any device that
is not `mps` or `cuda*` (the platform never runs a model on the CPU, design §3.1). It
resolves the snapshot with `huggingface_hub.snapshot_download(repo, revision=revision,
local_files_only=True)` and checks that the snapshot folder name IS the revision, so a
Hub id never silently loads `main` (`VoxCPM.from_pretrained("openbmb/VoxCPM2")`
downloads with no revision). Weights not on disk are a load error that names the fix
(`make models`, or `POST /models/download {slot: voice, key: voxcpm2}`); a film never
starts a 4.7 GB download. Then `VoxCPM.from_pretrained(<local path>,
load_denoiser=False, optimize=False, device=<device>, lora_config=LoRAConfig(**lora))`:

- `load_denoiser=False` — the ModelScope ZipEnhancer denoiser is never imported or loaded.
- `optimize=False` — `optimize=True` compiles only on CUDA with triton and adds a
  warm-up generate; it stays off until it is measured.
- The device is explicit: `"mps"` / `"cuda"` raises when it is not available — there is
  no fallback. On MPS the package forces float32. The child never sets
  `VOXCPM_MPS_DTYPE` or `PYTORCH_ENABLE_MPS_FALLBACK`.
- The `timestamps` extra is not installed. `warm()` runs one short zero-shot generate.

**The transcript sets the cloning mode.** Reference mode alone copies the TIMBRE and
leaves the pace to the model: measured on a Russian film, the copy spoke 6.58
syllables per voiced second against 4.90 in the person's own clip — a third faster.
So an example with a transcript (every `recorded` one) uses the package's "Ultimate
Cloning", which continues from the person's own speech and keeps "timbre, rhythm,
emotion, and style":

- **transcript present** — `build_prompt_cache(prompt_text=<transcript>,
  prompt_wav_path=<clip>, reference_wav_path=<clip>)` → `mode: "ref_continuation"`.
  The transcript is passed exactly as stored, as the package's own `generate` does.
- **no transcript** (`uploaded`) — `build_prompt_cache(reference_wav_path=<clip>)` →
  `mode: "reference"`, timbre only.

**Controls.** Each passage is whitespace-normalised and split at sentences into pieces
of ≤ 500 characters (`audio.silero_voice.split_text`); every piece is one
`generate_with_prompt_cache(target_text=piece, prompt_cache=<the voice's cache>,
min_len=2, max_len=4096, inference_timesteps=10, cfg_value=2.0, retry_badcase=True,
retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0)` — the package's own
`generate` defaults — and the pieces are joined directly. Same-voice passages are one
run (`audio/segments.py`). **There is no seed: 2.0.3 has no `seed` argument**, so the
child sets none.

These values are what every film uses. A reference may carry `controls` (design §5.1) with
its own `cfg_value` and `inference_timesteps`. The child lays them over a fresh copy of
`CONTROLS` for that token only, so `CONTROLS` never changes and a token beside it keeps
the fixed values. Only the `--sweep` and `--heldout` tests below send them.

**Conditioning blob.** Building the conditioning from a clip costs an AudioVAE
encode, so the child returns it for the caller to keep. A blob is `torch.save` of
`{"format": 1, "model": "voxcpm2", "revision", "voxcpm": <installed voxcpm version>,
"sha256": <clip sha256>, "mode": "reference" | "ref_continuation", "transcript":
<transcript or null>, "cache": {…}}`; everyone but the child
treats it as opaque bytes. On each request the child picks the stored blob for
`voxcpm2` + its revision and loads it with `torch.load(…, map_location="cpu",
weights_only=True)`. It is **restored** only when every field matches; otherwise it
is **rebuilt** and returned with the reason: `no stored conditioning`, `stored for
another revision`, `voxcpm version changed`, `clip changed`, `cloning mode changed`
(a reference-mode blob from before the transcript was used), or `transcript changed`. So a new pin, a
`voxcpm` upgrade or an edited example invalidates the blob, and the next call rebuilds
it. The reply carries `model="voxcpm2"` and `revision`, and the gateway refuses a reply
whose identity differs from the routed entry.

**Download.** `make models` (`scripts/fetch_models.py`) lists a pinned entry as
`"openbmb/VoxCPM2@<revision>"` and calls `snapshot_download(repo, revision=revision)`.
`voice` is an automatic slot, so on `mac` it pre-fetches VoxCPM2 at the pinned commit
(≈ 4.7 GB) beside Kokoro and Silero. `modelsvc/downloads.py` fetches it at the same
commit and reports it as `hf:openbmb/VoxCPM2@<revision>`.

**Unmeasured until the user's run.** The resident cost, the speed and the restore path
are proven only by `scripts/voice_check.py` in the user's own stack (CLAUDE.md
§ "Verify a model RUNS"). The Python 3.12 check so far is a syntax compile of the
2.0.3 sources and a resolve; the runtime import is part of that run.

**Dependencies and their licences.** `voxcpm==2.0.3` adds 67 packages per platform
(macOS 14 arm64 and manylinux_2_28 x86_64, CPython 3.12; the same set on both). All
have cp312/abi3/py3 wheels except `argbind==0.3.9` (sdist only). Every one passes the
permissive set (MIT, BSD, Apache-2.0, ISC, PSF-2.0, Zlib, …) except five approved
exceptions, listed below with exact versions:

- **`funasr` is removed.** voxcpm declares `funasr`, which pulls `kaldiio` 2.18.1 under
  an NTT evaluation-only licence (never approved). voxcpm 2.0.3 never imports
  `funasr`, so `pyproject.toml` carries `[tool.uv] override-dependencies = ["funasr;
  sys_platform == 'never'"]`: no install contains `funasr` or `kaldiio`. `uv.lock`
  still lists both, guarded by that marker.
- **`soxr==1.1.0` — LGPL-2.1-or-later.** Keep its four licence files (`COPYING.LGPL`,
  `LICENSE.txt`, `LICENSE-libsoxr.txt`, `LICENSE-PFFFT.txt`) with every copy. Give a
  prominent notice that the library is used and is under the LGPL, with the LGPL text.
  Ship the complete source (`soxr-1.1.0.tar.gz`) with the binary, or offer it from the
  same place or by a written offer valid for three years. It is one separate shared
  object (`soxr/soxr_ext.abi3.so`) loaded at run time, so a user can replace it; the
  distribution terms must permit modification for the user's own use and reverse
  engineering to debug it. PFFFT (BSD-3-Clause style): keep its notice, conditions and
  disclaimer in the documentation of a binary distribution; no NCAR/UCAR endorsement.
- **`orjson==3.13.0` — MPL-2.0 AND (Apache-2.0 OR MIT).** Keep the three licence files
  and notices. MPL-2.0 §3.2: whoever distributes the executable makes the source
  available, tells recipients how to get it (at no more than the cost of distribution),
  and does not limit their MPL rights in it. For the dual part pick one: MIT (keep the
  copyright and permission notice) or Apache-2.0 (give a copy of the licence; the wheel
  has no NOTICE file).
- **`cffi==2.1.1` — MIT-0.** No obligations; the wheel has no other licence file.
- **`pyahocorasick==2.3.1` — BSD-3-Clause and Public-Domain.** A binary distribution
  reproduces the copyright notice, conditions and disclaimer in its documentation; no
  endorsement with the authors' names. The public-domain part has no conditions.
- **`torchcodec==0.17.0` — BSD-3-Clause plus its bundled third-party licences.** Both
  wheels bundle dav1d (BSD-2-Clause), libavif (BSD-2-Clause, with IJG, Apache-2.0 and
  BSD-3-Clause parts), libjpeg-turbo (IJG, BSD-3-Clause, zlib), libpng (libpng-2.0),
  libwebp, libyuv (BSD-3-Clause) and zlib (Zlib). Obligations: BSD — reproduce the
  notice, conditions and disclaimer in the documentation of a binary distribution, and
  no endorsement with the authors' names; Apache-2.0 — give a copy of the licence and
  keep any NOTICE text; IJG — the documentation of an executable-only distribution
  states "this software is based in part on the work of the Independent JPEG Group";
  libpng-2.0 and Zlib — do not misrepresent the origin, mark altered source versions,
  and keep the notice in source distributions. The macOS wheel also bundles
  `torchcodec/.dylibs/libc++.1.0.dylib` with no licence file for it.
  - The **Linux** wheel also bundles `torchcodec.libs/libnvjpeg-*.so.13.0.1.86`, a
    proprietary NVIDIA redistributable under the bundled NVIDIA CUDA EULA
    (`third_party/LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt`), part of this approved
    exception. It is not a permissive licence. It allows commercial use: §1.1.1 grants
    use and the distribution of the parts listed as distributable, in object code,
    inside an application, and Attachment A (supplement §2.6) lists `libnvjpeg.so`.
    The obligations (§1.1.2): the application adds material functionality beyond the
    SDK; only the application accesses the library; the application's terms are
    consistent with the EULA; known or suspected non-compliant distribution is reported
    to NVIDIA in writing. Supplement §2.1: the SDK is licensed only for use in systems
    with NVIDIA GPUs. §1.2 also forbids reverse engineering, removing notices, and
    distributing the SDK as a stand-alone product. For context: the Linux resolve
    without voxcpm already ships NVIDIA CUDA libraries through torch (`nvidia-cublas`
    via `cuda-toolkit`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, …), whose wheels
    declare `License-Expression: LicenseRef-NVIDIA-Proprietary`.

The obligations above restate the licence texts in the selected wheels for unmodified
commercial use and redistribution; they are not legal advice.

### Voice baking — LoRA adapters

A saved voice can be **baked**: a LoRA adapter of VoxCPM2 is trained on the voice's
takes, and `/speak` uses it when a reference carries it (design §4.1, §5.1, §8.1). The
models service trains in a `voice_train` child (`audio/voxcpm_trainer.py:VoxCPMTrainer`)
and runs the bake job in `modelsvc/voice_bakes.py`. **`app` no longer bakes**: films
and tests speak from the voice's newest heard take of the language, arm B below (design
§11.1). In a blind test of 28 clips, the newest Russian take alone (arm B) beat the
stored baked v2 (arm C): likeness 4.75 vs 3.00, accepted 12 of 12 vs 6 of 12. The
user's `scripts/voice_bake_check.py` still bakes one read-only to measure it (below);
the stored voice models stay for it.

**The pinned training API** (`voxcpm` 2.0.3):

- LoRA layers exist only when the model is BUILT with a `LoRAConfig`; `VoxCPM.load_lora`
  raises otherwise. `LoRAConfig` defaults: `enable_lm=False, enable_dit=False,
  enable_proj=False, r=8, alpha=16, dropout=0.0`, targets `q_proj, v_proj, k_proj,
  o_proj` for the LM and the DiT. No LoRA layer sits in the AudioVAE.
- `VoxCPM2Model.from_local(path, optimize, training=True, device, lora_config)` is
  another model state: it skips the dtype cast, freezes the AudioVAE and every non-LoRA
  weight, keeps the VAE in float32, loads the checkpoint non-strict, and returns before
  `.to(device)`. So training cannot reuse the inference child.
- `forward(text_tokens, text_mask, audio_feats, audio_mask, loss_mask, position_ids,
  labels, *, progress, …)` returns `loss/diff` and `loss/stop`.
- `voxcpm.training` exports `Accelerator`, `load_audio_text_datasets` (a JSON-lines
  manifest read with HF `datasets`, audio cast to the VAE's sample rate),
  `build_dataloader` and `BatchProcessor` (packs a batch and VAE-encodes the audio
  under `no_grad`). `Accelerator(amp=False, seed=42)` seeds torch, numpy and random;
  its device is cuda, else mps, else the CPU **with no warning**.
- `get_lora_state_dict()` = every parameter whose name holds `lora_` (the export).
  `reset_lora_weights()` sets each A to a kaiming draw and each B to zero; it never
  touches a base weight. `load_lora_weights(path)` copies into ANY parameter whose name
  matches, base weights included, skips the rest in silence, and can raise halfway
  through on a shape that does not broadcast.
- `build_prompt_cache` encodes the clip with the AudioVAE only, so a stored
  conditioning blob does not depend on an adapter.

**Not in the wheel.** The package description (`METADATA`) names
`scripts/train_voxcpm_finetune.py` and `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`, but the wheel's `RECORD` lists no `scripts/` and no `conf/` path. So the package
pins no learning rate, step count or batch size, and the HF `datasets` path makes no
`text_ids` column (`HFVoxCPMDataset` needs one). The trainer here is our own, on
`voxcpm.training`.

**The recipe.** The LoRA shape is `AdapterSpec.lora` in the catalog: the package
defaults with `enable_lm` and `enable_dit` on and `enable_proj` off — the config the
package builds for itself when it loads LoRA weights. The loop values are this app's
defaults, a common starting point that no pinned source sets, and the user's bake run
measures them:

| | Default | A request may set |
|---|---|---|
| optimizer | AdamW over the LoRA weights | — |
| `learning_rate` | 1e-4 | `0 < x ≤ 1e-2` |
| `batch_size` | 1 | 1-8 |
| `weight_decay` | 0.01 | 0-1 |
| `max_grad_norm` | 1.0 | `0 < x ≤ 100` |
| `seed` | 42 (the package `Accelerator`'s own) | — |
| `steps` | `min(1000, max(200, 20 × takes))` | 1-5000 |
| loss | `loss/diff + loss/stop` | — |

One adapter per voice, from its takes in every language. The child, in order: it
refuses a device that is not `mps` or `cuda*`; sets `HF_DATASETS_CACHE` to
`<data_dir>/voice-bakes/hf-cache` and `HF_DATASETS_OFFLINE=1` before any `datasets`
import; resolves the pinned snapshot from disk only (as the voice child does); builds
the `Accelerator` FIRST, so the LoRA A draw is seeded; refuses when the accelerator's
device type is not the requested one; builds the model with `training=True` and the
catalog's `LoRAConfig`; and refuses unless every trainable parameter is a LoRA weight
(`check_trainable`). The text ids come from `ds.add_column("text_ids", …)` with the
model's `text_tokenizer`, never from a `ds.map` closure over the model: `datasets`
3.6.0 fingerprints a `map` transform with a pickler that would copy every tensor of
the model (read from the code, not seen in a run). The batches repeat as `while True:
yield from loader`, so each pass shuffles again (`itertools.cycle` would keep every
batch). A loss that is not a number fails the bake ("the training loss is not a number
at step N").

**The adapter file.** `adapter.safetensors` = `get_lora_state_dict()` on the CPU, with
string metadata: `format` `"ivms777-voxcpm2-lora-1"`, `model` `"voxcpm2"`, `revision`,
`voxcpm` (the installed version), `lora` (the sorted LoRA JSON), `dataset_sha256` (of
the sorted `<sha256>\t<language>\t<text>` lines of the takes), `config` (the loop
values as JSON) and `steps_done`. The bake status carries its sha256, size and key
count. A cancelled or failed bake saves no file.

**The switch in the voice child.** The child keeps one loaded adapter by sha256.
Before each run of a token: no adapter → `reset_lora_weights()` when one is loaded
(`none`); the loaded sha256 → nothing (`kept`); another one → it writes the blob to a
temporary `.safetensors` file and reads its metadata and each tensor's name and shape
with `safetensors.safe_open`, BEFORE any load. `adapter_mismatch` refuses a file that
is "not a voice model of this app", "made for another model revision" or "made with
another LoRA shape"; `tensor_mismatch` refuses names that are not EXACTLY the child's
LoRA names or a tensor of another shape ("the voice model holds N keys that are not
LoRA weights of this model", "the voice model misses N keys", "N tensors of the voice
model have another shape"). Only then: `reset_lora_weights()`, `load_lora_weights`,
and `load_mismatch` over the loaded and skipped keys (`loaded`). On ANY failure in
this step the child resets the LoRA weights and clears the kept sha256, then fails the
call with the reason, so no later request reads `kept` for a half-loaded model. The
pre-check is what keeps a base weight from ever being overwritten.

**The voice child is built with the LoRA config.** Its fourth arg is the catalog's
LoRA JSON, so it builds `VoxCPM.from_pretrained(…, lora_config=…)` and an adapter can
load without a rebuild. The LoRA layers, about 18 MB in float32 (an estimate: r 8 on
q/k/v/o of both LMs and the DiT ≈ 4.5 M parameters), sit inside `cost_mb` 10500. With
no adapter loaded every B is zero, so they add no adapter contribution, and plain
requests keep their inputs and controls; their extra matmuls still run and are not
measured. The runtime output and that overhead remain to be checked in the user's run.

**No new dependency.** The training path imports `datasets`, `argbind`, `pyyaml`,
`soundfile`, `safetensors` and `librosa`, which `uv.lock` already holds as `voxcpm`
dependencies: `datasets` 3.6.0 (`uv.lock:1106`), `argbind` 0.3.9 (`:281`), `pyyaml`
6.0.3 (`:3731`), `soundfile` 0.14.0 (`:4369`), `safetensors` 0.8.0 (`:4077`),
`librosa` 1.0.0 (`:2055`). `tensorboard` is not locked and not needed
(`TrainingTracker` runs with no writer, and the trainer does not use it).

**Unmeasured until the user's run.** The training cost (`train_cost_mb` 14000 is an
estimate), the seconds per step, the adapter size and the speed of the LoRA-built voice
child are not known. The likeness of a baked voice is not known either; the
speaker-similarity meter below gives evidence for it, and the ear judges it. In one
blind pilot on three pairs of the same voice model, the higher score lost to the user's
preference in two pairs. So the meter is an experimental diagnostic: never a clip
selector, an acceptance gate or a ranking of arms or architectures.
`scripts/voice_bake_check.py --voice NAME` measures them on the user's own stack: it opens the database read-only,
lists the voice's takes and where each take's words come from (`--list`): "heard by
<model>@<first 7 of revision>" (words that the app heard and stored), "shown
text, not heard" (a take stored before speech recognition: its words are the text it
was shown), "no usable words: <reason>", or "NO WORDS". It needs `--confirm-texts` and
words for every take (`--text ID=WORDS` gives them for this run). The script never
calls `/voice/transcribe` and writes no row, so a take has heard words only after the
app heard them; run the check after that. It bakes through `/voice/bakes`, keeps
the adapter only in its report folder, deletes the bake on every end, and speaks the
`voice_check` sentences three ways under random names: A the adapter + the prompt take,
B the prompt take alone (how films and tests speak now, on the LoRA-built child),
C the adapter alone (how films and tests spoke through a voice model). The prompt take of A and B is the
newest take of the sentence's language (`example_for`); `report.json`'s `prompt_takes`
records `{"example_id", "rule": "newest"}`. No take is chosen for films any more, so
the script reads no film choice.
`--steps N` and `--learning-rate X` set the training config for this run (else the
trainer's defaults), and `--out DIR` sets the output folder (default
`<data_dir>/voice-bake-check/<UTC stamp>/`). The console table is blind: per row the
sentence, the file (X, Y or Z), the duration, the dBFS and `ok` or `VOID`, and no
variant. `key.json` and `report.json` show which file is which variant, so listen to and
score every X/Y/Z file of a sentence BEFORE you open them. Its `report.json` holds the
bake status, the free memory before and at its lowest, the peak device memory, and per
call the variant, the adapter state, the RTF and the duration. The total recorded
minutes are printed beside the package authors' 5-10 minutes — their guidance, not a
promise. The pass rule is the script's own text: "PASS needs (§6.5 step 8): every
automatic check true; min_free_mb >= 1024 and macOS ended no process; seconds_per_step
<= 6.0; C (the adapter alone, as films will use it) at mean similarity >= B's mean +
0.5, mean naturalness >= 3.5, <= 1 wrong word per sentence. A and B only help to
diagnose."

**Measuring a stored voice model, with no bake.** One bake took 19617 s, so a check
that always bakes gives no number without hours of training. `--measure --repeats N
[--label NAME]` (N is 5 by default) skips the bake: it speaks through the STORED voice
model (`stored_model` + `read_voice_model`), as variants A, B and C, N times each. The
texts are the `voice_check` sentences, with RU-5, and RU-LONG, a new Russian passage over
the 500 characters of one synthesis piece. C is exactly what a film sends. Files are
`<text>-<repeat>-X/Y/Z.wav`, and `key.json` holds the order. A Ctrl-C ends the speaking,
and the report holds what was made.

Every file of a measure run and of a bake run is scored by the speaker-similarity meter
below, against the takes of its own language. RU-LONG is scored in fixed windows of 10 s
(`WINDOW_SECONDS`, a stated choice): a whole long clip can pass the 4 MiB limit, and windows
show drift across a piece boundary. The last window ends with the clip and may overlap the
one before, so every window has one length. A row's cosine is the mean of its windows only
when every window scored. If any window fails, the row has no cosine, its `score_error` is
"<n> of <m> windows could not be scored", and each window keeps its score or failure. The
script reads the database read-only, so it embeds the takes through `/voice/embed` and
never reads or writes `voice_take_embeddings`. The takes go first, so a meter that cannot
run stops the run before any speech. After a bake, a meter that cannot run leaves
`likeness.error` and the bake's result stands.

**Sweeping two generation settings.** `--voice NAME --sweep [--label NAME]` is its own
mode: no bake, arm C only (`use_clip: false`, the stored voice model), no production
setting changes. It speaks `SWEEP_TEXT`, one Russian sentence that no take, kit text or
check sentence holds, on the grid `cfg_value` 1.5, 2.0, 3.0 x `inference_timesteps` 10,
20. That is 6 cells x `SWEEP_REPEATS` 4 = 24 calls. Each repeat is one shuffled block of
all six cells, so a Ctrl-C stays balanced and drift spreads over the cells. All six send
`controls`, 2.0/10 too, so one path serves them. Before any call the run refuses a stored
model that is not version `SWEEP_MODEL_VERSION` (2), and a take whose words equal
`SWEEP_TEXT`.

File names are `S-01.wav` to `S-24.wav` from a second shuffle, so a name holds neither the
cell nor the call order; `key.json` holds the cell of each file. The reply must echo the
cell's `controls`. A missing or wrong echo stops the run at once: a stale service costs one
generation, not 24. The row stays, with its WAV, and the exit code is 1. A failed call is a
row and the run goes on, with no reroll. A Ctrl-C keeps what was made. `report.json` has its
own `frozen` block: the effective controls of the six cells, the text, the model identity,
the planned calls and the pass rule. `complete` is true only with all 24 clips valid, and
`cells` counts the valid clips per cell. The meter scores each file as a diagnostic: the score
stays in `calls`, the console prints none, and no ranking is made.

The pass rule is fixed before the listen. It is the `SWEEP_PASS_RULE` text of the script,
stored in the report. The user judges all 24 clips blind, on a local page with a random
order and no scores, and the key is joined only after submit. A pass earns only the question
of a held-out check; `CONTROLS` stays as it is until that check.

**The held-out check.** `--voice NAME --heldout [--label NAME]` is its own mode: no bake,
arm C only, no production setting changes. It tests one candidate from the sweep on new
words. `HELDOUT_TEXTS` are four short Russian texts, RU-HELD-1 to RU-HELD-4, and one long
text, RU-HELD-LONG, which crosses a synthesis piece boundary. No take, kit text, check
sentence, preview sentence, `LONG_PASSAGES` text or `SWEEP_TEXT` holds them. `HELDOUT_CELLS`
are the baseline 2.0/10 and the candidate 3.0/20, and both send `controls`. Each short text
runs `HELDOUT_SHORT_REPEATS` 3 times and the long text `HELDOUT_LONG_REPEATS` 2 times per
cell: 28 calls. Each repeat is one shuffled block of all text and cell pairs still due, so
the blocks hold 10, 10 and 8 calls, and both long repeats fall in the first 20 calls.
RU-HELD-LONG is scored in fixed windows, as RU-LONG is, but it stays out of `LONG_PASSAGES`,
so `--measure` never speaks it.

Before any call the run refuses: no Russian voice model, a stored model that is not version
`SWEEP_MODEL_VERSION` (2), a model whose sha256 is not `HELDOUT_MODEL_SHA256`, a take whose
words hold a held-out text (compared on letters and digits only), and an output folder that
is not empty. File names are `H-01.wav` to `H-28.wav` from a second shuffle, so a name holds
neither the cell, the text nor the call order. After the takes are embedded and before the
first call, `key.json` (all 28 planned files) and `frozen.json` go to disk. `frozen.json`
holds the five texts, both effective controls, the model identity, the planned rows, the
repeats, the pass rule and the wall budget. A killed run still leaves every file named.

The wall budget is `HELDOUT_WALL_SECONDS`, 1500 s. The clock starts just before the first
`/speak`. Each call gets the time that remains, both as the wait for its reply and as its
httpx timeout. httpx bounds each phase, not the total, so the script runs the call on a
worker thread and stops waiting there. The row is touched only on the main thread, and a
late reply is ignored. Any timeout marks the row `timed_out` and ends the run at once, even
with time left. A run stops with no replacement request when no time remains before a call,
and the clock is also read after each row. A failed call with time left is a row and the run
goes on, with no reroll. This is not a compute cutoff: `modelsvc` has no cancel path, so the
one call in flight may go on running on the service. The process exit waits for that worker
thread, which ends when the httpx timeout fires or the reply comes.

A wrong echo stops the run as in the sweep, and a Ctrl-C keeps what was made. `report.json`
is written at the end, after scoring, which runs outside the budget. It holds the keys of a
sweep report and `started`, `ended`, `elapsed_s`, `stopped_by_deadline` and `timed_out` (the
file). `complete` is true only with all 28 clips valid. The console prints no cell and no
score. The meter is a diagnostic only.

The pass rule is fixed before any held-out audio exists. It is the `HELDOUT_PASS_RULE` text
of the script, stored in `frozen.json` and the report. The user judges all 28 clips blind. A
pass earns only the question of changing `CONTROLS`; otherwise production stays unchanged.

**The longer bake.** `--voice NAME --longer-bake [--label NAME]` is its own mode. It bakes
one new voice model from the base model with `LONGER_BAKE_STEPS` (1000) steps; v2 had 520.
It sends only the steps, so every other loop value stays the trainer default, which is the
recipe of v2. The mode refuses `--steps` with another value and `--learning-rate`. It trains
on exactly the takes that v2 learned, as `voice_model_takes` records them. A take that v2 did
not learn is left out and named in `frozen.json`. Before the bake the run refuses: no Russian
voice model, a stored model that is not version `SWEEP_MODEL_VERSION` (2) with the sha256
`HELDOUT_MODEL_SHA256`, a take of v2 that is gone or being deleted, a take whose audio,
language or words differ from its fingerprint in `voice_model_takes`, and an output folder
that is not empty. `frozen.json` goes to disk before the bake starts: the config sent, the
recipe (the config of v2 with 1000 steps), the take ids, content hashes and fingerprints, the
dataset hash, the base revision, the seed and the watchdog limits. The new adapter is kept
only in the output folder as `adapter.safetensors`. The stored model and the database stay
unchanged, and there is no A/B/C speaking.

A watchdog follows this bake only. It requests the bake's cancel when 12 h
(`LONGER_BAKE_MAX_SECONDS`) pass since the start, when the status's `min_free_mb` falls below
1024, when `seconds_per_step` is above 45 after step 50, or when the bake fails. The trainer's
`seconds_per_step` is the mean since its first step. A Ctrl-C cancels as in every bake. The
cancel is cooperative, not a hard stop: the trainer stops at the next step boundary, so the
current step finishes first. The report records the trigger and the seconds since the start.
A bake that did not end `done` with all 1000 steps speaks nothing and exits 1 (130 after a
Ctrl-C). A bake that ends `done` after a cancel request keeps no adapter.

The service reports no start time, only `elapsed_s`. So the watchdog's start is the local
clock time when the first good status was requested minus its `elapsed_s` (for a new bake,
the reply of the start). The run computes it once and never resets it, and it keeps running
while polls fail. Each status request waits at most 60 s or the time left before 12 h,
whichever is less; when no time is left, the run sends no status request and requests the
cancel at once. A status request that times out or loses its connection prints one line and
is retried on the same bake at the next poll; the report counts these `poll_failures`. A
failed cancel request is retried at each poll until the bake ends. The free memory comes from
the status (`free_mb_before`, `min_free_mb`): no poll asks `/models`. A new start reads
`/models` once, before it, as every bake does. The service's bake folder is deleted only
after a status shows an end state (`done`, `failed` or `cancelled`); a run that leaves while
the bake still runs (a second Ctrl-C, an error) does not delete it and names it. Right after
the service accepts a new bake, the run writes its id to `bake.json` in the output folder.

**Attaching to a running longer bake.** `--longer-bake --attach BAKE_ID --out FOLDER` follows
a bake that already runs, for example after the client lost contact with it. It never starts
a bake. It makes the same refusals as a new run, but the output folder must hold this run's
`frozen.json` (equal to the one the run computes again) and nothing else, except a
`bake.json` that names the same id. Then it requests the bake's status once and refuses unless
it matches: `id` is BAKE_ID, `tag` is `{"check": NAME}`, `model` is `voice_model.model_key`,
`revision` is `base_revision`, `lora` is the stored v2's LoRA shape, `config` is `recipe`,
`clips` is the number of `takes`, and `dataset_sha256` is `dataset_sha256` of `frozen.json`.
Every refusal exits 2 before any write or cancel. The watchdog then follows the bake as above,
with its start from that status, so the 12 h count includes the time before the attach.
`report.json` records `attached` (true or false), `bake_id` and `poll_failures` on both paths.
`--attach` without `--longer-bake` is refused.

After a `done` bake, the run speaks the five `HELDOUT_TEXTS` through both models in arm C
(`use_clip: false`, no take). The candidate is the new adapter and the baseline is the stored
v2. No call sends `controls`, so both speak at the production `CONTROLS`. The plan is the
held-out plan with the two models in place of the two cells: 28 calls in blocks of 10, 10 and
8, and both long repeats fall in the first 20. File names are `L-01.wav` to `L-28.wav` from a
second shuffle. Before the first call `key.json` (file to model, text and repeat) goes to disk,
and `frozen.json` gains both adapter sha256 values, the planned rows, the pass rule and the
wall budget. The 1500 s budget, the timeouts, the Ctrl-C and the blind console are those of the
held-out check. The pass rule is the `LONGER_BAKE_PASS_RULE` text: clauses a-d of
`HELDOUT_PASS_RULE` with the new model as the candidate and v2 as the baseline. The meter and
the loss never decide. A pass earns only a separate question of adopting the new model for
this voice; a fail keeps v2 and ends the experiment. `report.json` holds the bake status, the
steps done, `seconds_per_step`, `peak_device_bytes`, the free memory before and the minimum,
the first and last loss, the stop (trigger and seconds), the `seconds_per_step <= 6.0`
threshold (reported, never a stop), the speaking fields of the held-out check and `complete`.

`report.json` holds:

- `results`: one entry per file, with nothing of VoxCPM2 — `label`, `arm`, `model`,
  `revision`, `text_id`, `repeat`, `cosine`, `seconds`, `windows`, `error`. A file that
  was not made or not scored keeps its reason and has no cosine: "not measured" is never
  a zero.
- `likeness`: mean, worst and spread with the failures, per arm, per text and arm, and
  per window position of a long text. Spread is highest minus lowest. The values per arm
  and per text cover complete files only: a file with a failed window counts as one
  failure. A window position keeps every window that scored. A scope with failures on
  either side has no delta in `--compare`: the two sides cover different files.
- `frozen`: the adapter sha256, the base revision, the mode of each arm, the texts,
  `CONTROLS`, and per language the reference: evaluator, pipeline, method, take ids and
  hashes, skipped takes, reference hash and the own range, the only scale.
- `calls`: the full rows of this model.

`format_summary` and the measure summary print no score, so the table stays blind until
the key is opened. The §6.5 pass rule stays by ear: a score is evidence, not a pass line.

**Comparing two runs.** `--compare OLD NEW` reads two `report.json` files and prints mean,
worst and spread per arm and per text for both, the deltas, and every frozen input that
differs (a text, a reference hash, the evaluator's pipeline). It reads only `label`,
`results` and `frozen`, so a report of another model or architecture compares with this
one if it writes those fields. Run it after you listen: it prints the scores.

**The app side — the voice model (no longer used by films and tests).** `app` starts no
bake any more; the text below describes the stored rows and a bake `app` started
before, which `reattach` still finishes or cleans up after a restart. A bake that `app` started first hears
the words of every take that has not been heard yet (the transcription stage,
"Hearing the words of a take — Whisper", below), stores them on the take, then sends
ALL the voice's takes with their heard words; a take whose heard words are unusable
stays in the voice, is left out and is named. The user types and confirms no words.
It tags the bake `{"source": "app",
"owner_id", "voice_id", "bake_key"}`, and stores the finished adapter as the voice's
**voice model**: one `voice_models` row per voice (design §6, `docs/data-model.md`)
with its version, the model key, the pinned revision, the LoRA JSON, the training
config, the dataset hash, the takes and seconds it learned from, the measured status,
and the safetensors file in `data_dir/voices`, content-addressed, removed only by the
voice outbox drain. `voice_model_takes` records each trained take's language, length
and fingerprint. Films and tests no longer send that adapter: they send arm B, the
newest heard take of the language with its words and no adapter, and each fails unless
the reply reports a conditioning (`built` or `restored`) for that take. The check
script's arm C still sends the adapter alone (`stored_model`, `model_reference`). `app`
never adopts, cancels or deletes a bake it did not start (the check script's, tagged
`{"check": NAME}`).

### Hearing the words of a take — Whisper

A film and a test send a take with its words, so VoxCPM2 continues from the person's
own speech (`ref_continuation`), and a bake trains on each take's audio AND its words
(the trainer tokenizes each take's text; `/voice/bakes` refuses a take whose text is
not 1-1000 characters after strip). `app` hears a take right after it is saved (the
hearing job, design §11.1).
A take is free speech, and the kit text is only a suggestion, so the words come from
local speech recognition in the models service, never from the user and never from
the kit text (design §4.1, §5.1, §8.1, §11.1).

- **The model.** `whisper-large-v3-turbo` = `openai/whisper-large-v3-turbo` @
  `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. The weights are **MIT** (the Hub's
  `cardData.license`; the README says the same), not gated. 808,878,080 parameters;
  one weight file of 1,617,824,864 bytes (F16), and all 13 files are 1,622,466,054
  bytes (`size_mb` 1547). A pruned large-v3 with 4 decoder layers in place of 32: the
  model card says it is much faster, at a minor quality loss. Declared in
  `catalog.TRANSCRIBERS` with `transcriber_for(profile)`, outside `CATALOG` and `SLOTS`:
  no slot, no default, no settings row.
- **Where it runs.** On mac and cloud (`profiles=("mac", "cloud")`), every profile
  that clones, so each take is heard right after it is saved. One `asr`
  child in the models service (`audio/whisper_asr.py:WhisperASR`), loaded in
  **float32 on MPS or CUDA** (no float16 run was measured here), from the pinned
  snapshot on disk only. It is counted in the voice gate, runs at BATCH, and a bake
  unloads it before training, so it never shares memory with the training child.
- **The input.** One take per `POST /voice/transcribe` call. A take is a mono PCM-16
  WAV at 48 kHz, 5-30 s, so it fits Whisper's one 30 s window in one pass. The child
  resamples it to Whisper's 16 kHz exactly (48000 / 16000 = 3,
  `scipy.signal.resample_poly(x, 1, 3)`), and decodes with the take's language forced
  (`en` or `ru`, `task="transcribe"`); nothing detects the language. It returns the
  text as decoded. `app` alone judges it: words with no letter and no digit (blank, or
  symbols only) give "the speech recognition found no words in it", and more than 1000
  characters give "it holds more than 1000 characters of words". Such a take stays,
  shows the reason, and no film, test or bake speaks from it or learns from it.
- **No new dependency.** `transformers` 5.15.0 (the `models` extra, already locked)
  holds `WhisperForConditionalGeneration` and `WhisperProcessor`. `scipy` 1.18.0 is
  locked and installed as a dependency of `librosa` 1.0.0, which `voxcpm` needs. No
  package and no licence exception is added.
- **The cost is an estimate; not measured.** `cost_mb` 4000 with `cost_measured=False`
  = 3086 MiB of float32 weights (808,878,080 × 4 B) + an unmeasured allowance for the
  encoder activations of one 30 s window and the runtime. `IVMS777_MODEL_COST_MB
  '{"asr": <measured>}'` overrides it. When a transcribe call loads the child, the
  service logs `asr <key>@<first 7 of revision> loaded by this request: free_mb
  <before> → <after>, net change <N> MB; evicted: <units, or none>`. That line reports
  the net free-memory change of that request, other units it evicted included; it is
  not a cost. A cost is pinned only from a load that evicts nothing, measured as
  design §8.1 requires.
- **Accuracy.** No accuracy figure is claimed. Words it hears wrongly are sent as
  heard, and nobody corrects them: in the blind test 2 of 14 sample clips had wrong
  words. A take is heard once, also after a later revision.

### The speaker-similarity meter — ECAPA-TDNN

The user needs a number that compares generated speech with the recordings of a saved
voice. A speaker-verification model gives one. It only measures: synthesis, the voice
model and films do not change (design §4.1, §5.1 `/voice/embed`, §8.1, §11.1).

- **The model.** `ecapa-voxceleb` = `speechbrain/spkrec-ecapa-voxceleb` @
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. The weights are **Apache-2.0** (the Hub's
  `cardData.license`), not gated. ECAPA-TDNN, a 192-value embedding, 16 kHz mono input.
  All 10 files of the pinned snapshot are 89,133,738 bytes (`size_mb` 85):
  `embedding_model.ckpt` 83,316,686 B, `classifier.ckpt` 5,534,328 B,
  `label_encoder.txt` 128,619 B, and small files. The whole snapshot is fetched, because
  the pinned `hyperparams.yaml` pretrainer also loads `classifier.ckpt` and
  `label_encoder.txt`. Declared in `catalog.SPEAKER_EMBEDDERS` with
  `speaker_embedder_for(profile)`, outside `CATALOG` and `SLOTS`: no slot, no default,
  no settings row.
- **Where it runs.** On mac only (`profiles=("mac",)`). One `speaker_embed` child in the
  models service (`audio/speaker_embed.py:SpeakerEmbedder`), from the pinned snapshot on
  disk only. A device that is not `mps` or `cuda*` is a load error: there is no CPU
  fallback. It is counted in the voice gate, runs at BATCH, and a bake unloads it before
  training. Elsewhere `POST /voice/embed` answers 503, and the test says "not measured".
- **The input.** One clip per `POST /voice/embed` call. The endpoint takes a mono PCM-16
  WAV at 48 kHz of at most 4 MiB (about 43 s), with no 30 s cap: takes and VoxCPM2's
  output are this format. The child accepts any rate from 16000 Hz and resamples to
  16 kHz (`scipy.signal.resample_poly`). A clip with under 1.0 s of 30 ms frames at RMS
  0.01 or more is refused with its reason (`MIN_VOICED_SECONDS`, `VOICED_RMS`): a stated
  choice, not calibrated, and an energy check rather than voice detection.
- **The reply.** The raw vector, with `revision`, `seconds`, `elapsed_s` and `pipeline`
  (`audio/speaker_embed.py:PIPELINE`, an integer that goes up when the preprocessing
  changes). A vector that is not finite or has a zero norm is a model error. `app`
  normalises and compares (design §11.1).
- **The scale.** The voice's own leave-one-out take range only. The page reads "Speaker
  similarity 0.31 (cosine) · your own takes 0.72-0.85". No threshold ships, and the
  number is never a percent or a probability.
- **The cost is an estimate; not measured.** `cost_mb` 1000 with `cost_measured=False` =
  85 MiB of files + an unmeasured allowance for the torch runtime and the activations of
  a 43 s clip. `IVMS777_MODEL_COST_MB '{"speaker_embed": <measured>}'` overrides it.
- **Unknown until measured.** The run on MPS (one real call must give 192 finite values
  with a nonzero norm, and the same clip twice must agree within a declared tolerance);
  that proves runtime only. The Russian accuracy of a VoxCeleb model. **With no second
  person, the meter's separation of two speakers is not proven on this user's data.** A
  short test may score lower for its length alone, because the own range comes from
  5-30 s takes.

**Dependencies and their licences.** `speechbrain==1.1.2` (the `models` extra) adds 5
packages to an install. `uv lock` adds 4 of them; `sentencepiece` was already locked and
not installed. The licences below are read from each package's dist-info after install.
All have py3 wheels, or cp312 wheels for macOS arm64 and manylinux x86_64 and aarch64.
Every one passes the permissive set, so there is no new exception:

| Package | Version | Licence | Licence text in the wheel |
|---|---|---|---|
| `speechbrain` | 1.1.2 | Apache-2.0 | `licenses/LICENSE` |
| `hyperpyyaml` | 1.2.3 | Apache-2.0 | `licenses/LICENSE` |
| `ruamel-yaml` | 0.18.17 | MIT | `licenses/LICENSE` |
| `ruamel-yaml-clib` | 0.2.15 | MIT | `licenses/LICENSE` |
| `sentencepiece` | 0.2.2 | Apache-2.0 (`License-Expression`) | none: a copy that is redistributed must add the Apache-2.0 text |

`speechbrain` also requires `torchaudio`, `joblib`, `requests`, `scipy`, `soundfile`,
`tqdm` and `huggingface_hub`, which `uv.lock` already holds.

## Switching a slot

`app` owns the choice (`app_settings`, one row per slot) and resolves
**stored → env override → profile default**. On confirm, in one transaction:
write the choice → rebuild `photo_vec` if the new `image_embed` `dim` differs →
`jobs.reprocess()` over the slot's invalidated range — `image_embed`:
`embed`→`taxonomy`, `text_embed`: `caption_embed`, `caption`:
`caption`→`caption_embed`, `planner`: nothing. Then `PUT /models/slots` to the
service, which evicts the outgoing resident and re-registers the slot; the next op
loads the new model.

The `models` service has no DB. It boots on profile defaults and reports its slots
with a `generation`; on the resource-bar poll `app` compares the reported switchable
slots (`catalog.is_switchable`) with its stored selection and re-pushes only those when
they differ, so a `models` restart converges without a shared store. The automatic
`voice` slot is never compared or pushed — the service routes it per request.

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

**`make up` downloads every model the app needs before it starts anything**
(`make models` → `scripts/fetch_models.py`): every selected slot, plus every entry
of an AUTOMATIC slot (`voice` routes by language, so both Kokoro and Silero), and,
each at its pinned revision, Whisper on every profile that offers it (mac and cloud,
`transcriber_for`, about 1.6 GB, once) and the speaker-similarity model on every
profile that offers it (mac, `speaker_embedder_for`, about 89 MB, once). It uses the same `IVMS777_DATA_DIR`/`IVMS777_PROFILE` defaults as `up`, so a URL model
(Silero) lands in the `DATA_DIR/models` the `models` service reads. **A failed
download stops `make up`** with each failed model and its reason — it used to be
swallowed, and a fetch aimed at the container path `/data` failed on every Mac
while the app started anyway with no Russian voice. Already-present weights are
skipped, so a rerun costs nothing.

**Kokoro's espeak data must sit on a short path.** espeak-ng keeps its data path in
a 160-byte buffer; a longer one (a worktree venv put the bundled `espeakng_loader`
data at 166 characters) is dropped, the library falls back to its build machine's
path and `exit(1)`s the voice child. `audio/espeak_path.py` hands espeak a short real
copy of the data instead (a symlink is resolved back to the long path). The spaCy
English model misaki loads is a declared `models` dependency, so `uv sync` keeps it.

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
