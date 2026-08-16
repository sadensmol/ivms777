# Plan 16 — Collapse text + vision onto a single Gemma 4 via llama.cpp (Jetson)

Status: **VIABLE — gate 1 now PASSES via a source-built `sm_87` llama.cpp
(2026-08-16).** Stock Ollama vision failed (CPU projector, 211 s), but a
source-built llama.cpp runs the vision projector **on the GPU** and captions
`gemma4-E2B` in **0.57 s/image**. Snapshot derived from the design
(§3.1/§4/§5.1/§8.1) and the 2026-08-16 Jetson benchmark session. Supersedes the
in-process captioner approach of `14-jetson-inprocess-captioner.md`.

## Validation results (2026-08-16, MAXN)

- **Gate 1 (vision on GPU) — FAILED on Ollama, PASSES on source-built llama.cpp.**
  - *Ollama:* `gemma4:e2b` = **211 s, empty**; `qwen2.5vl:3b` = **190 s** (correct
    caption). Both CPU projector, unusable for bulk.
  - *Source-built `sm_87` llama.cpp* (`-DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES=87`), projector **on GPU** (default offload),
    `-ngl 99 --flash-attn on --jinja -c 4096`:
    - **`gemma4-E2B` image encode = 0.57 s/image**, correct detailed caption.
    - `Qwen2.5-VL-3B` = 14 s/image (heavier: ~2048 image tokens).
    - CPU projector (`--no-mmproj-offload`): qwen = 149 s → GPU ~10×; gemma ~370×
      vs Ollama.
    - Built inside `ghcr.io/nvidia-ai-iot/ollama:...cu130` (has nvcc/cmake); the
      cu130 binary runs on the cu132 driver, no crash.
    - **gemma4 needs `--jinja`** (its chat template aborts otherwise) — the cause
      of Ollama's earlier "empty output".
- **Gate 2 (memory) — PASS (verify coexistence).** `gemma4-E2B` Q4_K_M ≈ 2.9 GB +
  mmproj ≈ 0.9 GB.
- **Gate 3 (quality)** — first read good (detailed correct captions); full
  bake-off vs Qwen2.5-VL on real photos still to run.
- **Text (bonus).** `gemma4:e2b` = **26.1 tok/s** (faster + smaller than
  `qwen2.5:3b` 23.5).

**Conclusion:** the single-model unification is achievable — **not on Ollama**
(no GPU vision path) but on a **source-built `sm_87` llama.cpp server**. One
`llama-server` running `gemma4-E2B` (+ mmproj) serves **chat (26 tok/s) and
captioning (0.57 s/image)**, both on the GPU.

## Revised direction (what ships)

1. **Add a Jetson `llama.cpp` build** (in `Dockerfile.models.jetson` or a sidecar
   image): clone ggml-org/llama.cpp, `cmake -DGGML_CUDA=ON
   -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=ON`, build `llama-server`.
2. **Run `llama-server`** with `gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-*-f16.gguf`,
   `-ngl 99 --flash-attn on --jinja -c 4096`, OpenAI-compatible endpoint.
3. **Point the `models` service text + caption backends at that `llama-server`**
   (single inference gateway, §5.1) — drop the in-process transformers captioner,
   `bitsandbytes`/`accelerate`, and the caption↔planner residency swap. This also
   lets us **drop Ollama on jetson entirely** (one server for text + vision).
4. **Design edits, same turn as code**: §3.1 profile table (jetson inference =
   `llama.cpp`, one Gemma for text+vision), §4, §5.1, §8.1, and the §5 mermaid.
5. **MAXN boot-persistence** service.

Open items: caption-quality bake-off (gemma4-E2B vs Qwen2.5-VL); memory
coexistence (llama-server + SigLIP in ~5.8 GB); Ollama-vs-llama.cpp for text.

## Goal

On `jetson`, run **one** multimodal model — `gemma4` on `ollama:latest` — for
**both** the planner/chat text role *and* captioning, replacing:

- `qwen2.5:3b` (planner/chat, Ollama), and
- `Qwen2.5-VL-3B` 4-bit (captioning, **in-process** via `transformers`).

Net simplification if adopted:

- Delete the in-process caption backend (`modelsvc/backends/caption_backend.py`,
  `captioning.VLMCaptioner`) and its jetson-only deps (`bitsandbytes`,
  `accelerate`, `gcc`/Triton headers) from `Dockerfile.models.jetson`.
- Remove the caption↔planner residency swap on jetson (§8.1): captioning leaves
  the `models` process entirely, so the only in-process model left there is
  SigLIP. No swap between two heavy in-process models.
- One backend/transport for all generation: Ollama serves text **and** vision,
  reached only through the `models` service (keeps the one-gateway rule).

**SigLIP stays** — Gemma does not replace it. SigLIP remains the image/text
*embedding* + zero-shot tag model (§9), in-process in `models`.

## Why now (findings, 2026-08-16)

- Setting **MAXN_SUPER** unthrottled Ollama from 3.0 → **23.5 tok/s** on
  `qwen2.5:3b` (llama.cpp fused GGUF kernels, bandwidth-bound). Usable chat, no
  code change. (Design §3.1 benchmark block.)
- NVIDIA confirmed the Orin `sm_87` (cc=870) fix is in recent Ollama — generic
  `ollama:latest` runs text on the Orin GPU natively; no custom build needed.
- `gemma4` is an official **multimodal** Ollama library model (`e2b`/`e4b`/12b/
  26b/31b; vision on every variant) and is the design's already-named intended
  captioner (`gemma4:e4b`, §3.1/§4).
- No prebuilt r39.2/cu132 Ollama exists; building one is painful (jetson-
  containers #1661). So the viable path is the **stock `ollama:latest`**, which
  already works for text — this plan just adds Gemma for vision.

## Gates (ALL must pass before adopting — do not implement until then)

1. **Vision runs on the Orin GPU under Ollama** — `gemma4` caption request does
   **not** deadlock on CLIP-on-CPU (the failure that pushed captioning
   in-process for qwen-VL). `ollama ps` shows GPU; request returns.
2. **Memory fits the 8 GB budget** — `gemma4:e2b` (~3 GB) resident + SigLIP
   (~1 GB) + ~1.5 GB baseline ≈ 5.5 GB ✓. `e4b` (~5 GB) is borderline (≈7.5 GB
   with SigLIP) — default to **e2b** unless a memory-coexistence test clears e4b.
3. **Quality is good enough on real photos** — a caption bake-off vs the current
   `Qwen2.5-VL-3B`, and a planner/chat sanity check vs `qwen2.5:3b`. Gemma must
   not regress caption tags/search or the agentic planner.

(Gate 1+2+3 first read comes from the `gemma4:e2b` validation run started
2026-08-16 — text tok/s + a real vision caption at MAXN. Record its result here.)

## Steps (once gates pass)

1. `config.py` jetson profile: set the planner and captioner to a single
   `gemma4:e2b` (or `e4b` if memory clears); set `caption_backend = "ollama"` on
   jetson so `modelsvc/backends/__init__.py` builds the Ollama caption adapter,
   not the in-process one.
2. Delete the in-process caption path: `modelsvc/backends/caption_backend.py`
   (in-process branch), `captioning.VLMCaptioner`, and their tests; drop
   `bitsandbytes`/`accelerate`/`gcc` from `Dockerfile.models.jetson`.
3. Remove the jetson caption↔planner residency swap in `modelsvc/residency.py`
   (SigLIP is the only in-process heavy model left).
4. `make run-jetson` pulls `gemma4:e2b` into the in-container Ollama (it is now a
   real Ollama tag on jetson, unlike the old HF `caption_model_id`).
5. **Design edits, same turn as code**: §3.1 profile table (jetson caption →
   Ollama `gemma4`, one model), §4 (captioner), §5.1 (no in-process VLM), §8.1
   (residency = SigLIP only), and the **§5 mermaid** (drop the in-process caption
   VLM box; Ollama now serves caption + text).
6. Add a **MAXN persistence** boot unit (`nvpmodel -m 2` persists; `jetson_clocks`
   does not) so performance survives reboot.

## Risks / rollback

- If gate 1 fails (vision deadlocks on Ollama) → keep the in-process captioner;
  this plan is void, revisit only with a source-built `sm_87` llama.cpp (NVIDIA's
  Gemma-4 demo path).
- If gate 3 fails (quality regression) → keep `qwen2.5:3b` for text and the
  in-process `Qwen2.5-VL-3B` for caption; do not unify.
- Rollback is config-only until step 2 deletes code; keep the in-process backend
  behind `caption_backend` until Gemma is proven in production.

## Not in scope

- Tuned-vLLM (36.3 tok/s) as the text engine — recorded in the design benchmark
  block as the faster-but-memory-heavier alternative; parked unless a memory
  coexistence test clears it alongside SigLIP + caption.
- Building a custom `sm_87` Ollama/llama.cpp — only if Ollama vision is
  unworkable and captioning must be reclaimed onto llama.cpp.
