# Plan 16 — Gemma 4 on GPU llama.cpp everywhere; drop Ollama (mac + jetson)

Status: **APPROVED 2026-08-16.** One model (`gemma4-E2B`) for **text + vision**,
served by **llama.cpp `llama-server` on the GPU on every profile**; **Ollama
removed entirely**; the in-process `transformers` captioner removed. Supersedes
`14-jetson-inprocess-captioner.md`. Design sections to change at implementation:
§3.1 (profile table), §4 (captioner), §5.1 (backend + gateway), §8.1 (residency),
and the §5 mermaid.

## The decision (final)

- **One model:** `gemma4-E2B` — `gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-*-f16.gguf`
  (unsloth). Serves the planner/chat text role **and** captioning.
- **One engine, GPU only:** llama.cpp **`llama-server`** (OpenAI-compatible API).
  - **mac** — native **host** build with **Metal** (`-DGGML_METAL=ON`). Docker on
    macOS has no GPU (same reason Ollama ran on the host, §3.1), so `llama-server`
    runs on the host and the `app`/`models` containers reach it at
    `host.docker.internal:8080`.
  - **jetson** — **containerized** build with **CUDA `sm_87`**
    (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=ON`), run as a
    compose service with `runtime: nvidia`. (The Jetson host has no
    `nvcc`/`cmake`; build in the image — consistent with the containerized design.
    Verified: the cu130 build env runs fine on the cu132 driver.)
- **No Ollama** on either platform. **No in-process `transformers` VLM**, no
  `bitsandbytes`/`accelerate`. The `models` service is the single gateway (§5.1)
  and calls `llama-server` over **OpenAI `/v1/chat/completions` for both text and
  vision** (image via `image_url` data-URI).
- **SigLIP stays** in the `models` service for embeddings + zero-shot tags (§9).
  Caveat: in-container SigLIP on **mac** stays **CPU** (Docker has no Metal); on
  **jetson** it stays CUDA (cu132 torch). "GPU-only" applies to the gemma
  inference; moving mac SigLIP onto Metal would require a host process too — out
  of scope (open item).

## Required `llama-server` flags (both platforms)

```
llama-server -m gemma-4-E2B-it-Q4_K_M.gguf --mmproj mmproj-gemma4-e2b-f16.gguf \
  -ngl 99 --flash-attn on --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -c 4096 --host 0.0.0.0 --port 8080
```

- **`--chat-template-kwargs '{"enable_thinking":false}'` is mandatory** — without
  it gemma4 dumps chain-of-thought instead of a caption (`--reasoning-budget 0` is
  not enough). Verified 2026-08-16.
- **`--jinja`** required (gemma4 chat template).
- Caption requests send a **system prompt**: "reply with ONE detailed paragraph —
  objects, people, visible text/numbers, setting; no analysis or reasoning."
- jetson: run under **MAXN_SUPER** (`nvpmodel -m 2 && jetson_clocks`) — ~8× vs the
  25 W default.

## Measured basis (2026-08-16, MAXN, sm_87 llama.cpp)

| `gemma4-E2B` (GPU) | value |
|---|---|
| Text | **30 tok/s** |
| Vision (image encode) | **0.57 s** |
| Caption end-to-end | **~5–6 s/image** |
| Quality | matches Qwen2.5-VL (reads in-image text, fine detail; caught a faint watermark Qwen missed); more grounded/less hallucinatory. Bake-off on dog/bus/people at ~2–4× Qwen's speed. |

(`Qwen2.5-VL-3B` for reference: text 23.5 tok/s, vision 14 s encode, ~20 s/img;
slightly better only on dense-text OCR.) Full numbers in design §3.1.

## Implementation steps

1. **llama.cpp build**
   - *jetson:* `Dockerfile.llamacpp.jetson` (or a build stage in
     `Dockerfile.models.jetson`) — clone `ggml-org/llama.cpp`, `cmake -DGGML_CUDA=ON
     -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`,
     build `llama-server`; ship the binary + `libggml-cuda.so` etc.
   - *mac:* README build recipe (native Metal) — see step 8. Not containerized.
2. **GGUFs**: fetch `gemma-4-E2B-it-Q4_K_M.gguf` (unsloth) + `mmproj-F16.gguf`
   (unsloth `gemma-4-E2B-it-GGUF`) into a mounted models volume/dir; `make`
   targets pull them (idempotent).
3. **compose**
   - `compose.jetson.yaml`: **remove** the `inference: ollama/ollama` service;
     add `inference: llama-server` (built image, `runtime: nvidia`, the flags
     above, GGUF volume). `models` depends on it.
   - `compose.mac.yaml`: point `models` at `host.docker.internal:8080`; no
     inference container (host `llama-server`).
4. **`config.py`**: mac + jetson inference = the `llama-server` OpenAI endpoint;
   planner/chat **and** caption model = `gemma4-E2B`; `caption_backend` → the
   OpenAI/llama path; delete the `inprocess` VLM branch + `caption_model_id` (HF).
5. **`modelsvc`**: one OpenAI client for text **and** vision — caption backend
   POSTs `/v1/chat/completions` with an `image_url`; **delete**
   `backends/caption_backend.py` in-process branch, `captioning.VLMCaptioner`, the
   `bitsandbytes`/`accelerate` deps, and the caption↔planner swap in
   `residency.py` (SigLIP is the only in-process heavy model left).
6. **Makefile**
   - `make up` (mac): ensure host `llama-server` is built + running (a
     `make llama-mac` helper builds/starts it), then `docker compose … up`.
   - `make run-jetson`: build the llama.cpp image, pull GGUFs, **set MAXN**
     (`ssh … nvpmodel -m 2 && jetson_clocks`), `docker compose … up`. All the
     `llama-server` optimization flags live in the compose service, not ad-hoc.
   - Document a **MAXN boot-persistence** unit (`jetson_clocks` doesn't survive
     reboot).
7. **`Dockerfile.models.jetson`**: drop the caption-VLM bits
   (`bitsandbytes`/`accelerate`, the Triton `gcc`); keep cu132 torch for SigLIP.
8. **`README.md`**: add **"Build llama.cpp for Mac (Metal)"** —
   `brew install cmake`; `git clone https://github.com/ggml-org/llama.cpp`;
   `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build
   build -j`; download the gemma GGUF + mmproj; the `make up` flow expects
   `llama-server` on `:8080`. Note the jetson build is automatic via `make
   run-jetson`.
9. **Design edits (same turn as code)**: §3.1 table (mac/jetson inference =
   llama.cpp, one `gemma4-E2B` for text+vision, GPU), §4, §5.1 (llama-server is
   the only backend; no Ollama; no in-process VLM), §8.1 (residency = SigLIP
   only), §5 mermaid (Ollama box → `llama.cpp llama-server`, serving caption +
   text).

## Open items / out of scope

- **cloud** profile: unchanged here (it uses vLLM, never Ollama). Decide later
  whether to also move cloud to gemma/llama.cpp or keep vLLM.
- **mac SigLIP on GPU**: not possible in a mac container (no Metal in Docker);
  would need a host process. Out of scope unless required.
- Widen the **caption-quality bake-off** (dense text, low light, many objects)
  before the first bulk ingest.
- MAXN boot-persistence service for the Jetson.

## Validation results that back this (2026-08-16, MAXN)

- **Vision on GPU works only via source-built `sm_87` llama.cpp** (projector on
  GPU). Ollama runs the projector on **CPU** (gemma 211 s/empty, qwen 190 s) —
  unusable. GPU offload: gemma **0.57 s** encode, qwen **14 s** (CPU projector
  149 s → ~10–370× win).
- **Text on GPU:** gemma4-E2B **30 tok/s** (> Ollama's 26.1), qwen-VL 23.5.
- **`enable_thinking=false`** turns gemma's broken thinking-dump into clean,
  Qwen-quality captions (reads bus text/route/"cero emisiones", counts people,
  even wrought-iron balconies + a faint watermark).
- The cu130 build env produces a binary that runs on the cu132 driver (no crash).
