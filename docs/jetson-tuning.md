# Jetson Orin Nano — tuning & benchmark reference

This is the **empirical reference** for the `jetson` deploy profile: the one
target device's facts, the power-mode lever, the measured single-stream and
vision benchmarks, and the memory budget behind them. It is measurement, not
design — the deployment *design* decisions (profile table, why llama.cpp over
vLLM, the one-model rule, the Jetson image) live in `docs/design.md` §3.1/§4/§5.
When a number here and the running board disagree, re-measure; this file is only
as current as its last verification date.

## Memory budget — where the 8 GB goes

The Orin Nano Super has 8 GB shared between CPU and GPU at 102 GB/s (~7.4 GB
usable after firmware). A **fixed ~1.5 GB baseline is gone before we load
anything** — measured on an idle board (JetPack 7.2, no containers): ~1.1 GB
kernel + iGPU/firmware carveout reserved by L4T at boot, ~155 MB kernel slab,
~230 MB system daemons (dockerd, containerd, systemd). This baseline is **not
reclaimable** — it is set in the device tree, not held by any process, so killing
services claws back only tens of MB (headless mode a few hundred at most; not
worth chasing). That leaves roughly **5.8 GB** as the real model budget.

Since plan 16 the pieces that share it are: the gemma4-E2B GGUF on `llama-server`
(~2 GB weights + a CUDA context), SigLIP (~1.6 GB) inside the `models` service,
and the small in-process caption-text embedder (nomic, ~0.3 GB) — all comfortably
inside ~5.8 GB. The one heavy in-process model in the `models` service is now
**SigLIP alone**: the caption VLM that used to share the GPU with it is gone
(captioning is a remote call to `llama-server`). So the residency manager (design
§8.1, `modelsvc/residency.py`) collapses to an **ensure-loaded** guard —
`use("siglip", HIGH)` loads SigLIP once and reports it for the resource bar; there
is nothing to evict, swap, or preempt it against. At MAXN_SUPER expect ~5–6
s/photo for captioning (below).

## Device facts (`lockbox-nv`) — verified on the live board 2026-08-16

The one target device, for tuning reference:

| Field | Value |
|---|---|
| Board | Jetson Orin Nano **Super** (Engineering Reference Dev Kit) |
| SoC / GPU | Ampere iGPU, **compute capability `sm_87`**, 1024 CUDA cores, 32 tensor cores; GPU clock max **918 MHz** |
| CPU | 6-core Arm Cortex-**A78AE**, max **1.728 GHz** (L1 384 KiB, L2 1.5 MiB, L3 4 MiB) |
| AI perf (spec) | **67 INT8 TOPS** — relevant to *prefill / batched / vision* (compute-bound), **not** single-stream chat decode |
| Memory | **8 GB** 128-bit LPDDR5, **102 GB/s** peak (EMC max **3199 MHz**); ~7.4 GB usable, **~5.8 GB** model budget after the fixed baseline |
| Storage | NVMe **Crucial P310** (`CT1000P310SSD8`) 1 TB; `/` on a 915 GB ext4, ~22% used |
| OS | **Ubuntu 24.04.4 LTS** (noble) |
| L4T / JetPack | **L4T r39.2** (`nvidia-l4t-core 39.2.0`) = **JetPack 7.2** |
| Kernel | **6.8.12-1021-tegra** aarch64, PREEMPT |
| CUDA on host | **runtime only** — `libcuda.so.1` from `/opt/nvidia/l4t-gpu-libs/nvgpu`; **no toolkit, no `nvcc`, no `cmake`/`g++`**. All CUDA/toolchain (cu132) is per-container. |
| Container CUDA | **CUDA 13.2** (cu132) inside images |
| Docker | **29.6.2**, `default-runtime = nvidia` (so `runtime: nvidia` is implicit), overlay2, cgroup v2 |
| Python (host) | 3.12.3 |
| Power modes | `nvpmodel` IDs: **`0`=15W, `1`=25W, `2`=MAXN_SUPER**. `jetson_clocks` then locks max clocks. |
| Access | `lockbox@192.168.100.8` (`lockbox-nv.local`); **passwordless SSH**, but **`sudo` needs a password** |

**Power mode is the single dominant lever — set MAXN_SUPER.** The board ships in
**`25W` (ID 1)**, which throttles the GPU to ~306 MHz and the memory controller
(EMC) to ~2133 MHz (~68 GB/s), even though thermals are cool (~49 °C).
`sudo nvpmodel -m 2 && sudo jetson_clocks` switches to **MAXN_SUPER**: GPU locks
to **1020 MHz** and EMC to **3199 MHz** (full 102 GB/s). The `nvpmodel` mode
persists across reboot; **`jetson_clocks` does not** — a small boot service must
re-apply it. Fix this before blaming any stack.

## Single-stream decode benchmark (verified 2026-08-16)

`qwen2.5:3b` 4-bit, single-stream decode, server-reported tok/s:

| Engine / config (single-stream) | 25 W | **MAXN_SUPER** | scaling |
|---|---|---|---|
| **vLLM `0.22` tuned — AWQ-Marlin + CUDA graphs** | — | **36.3** | winner |
| **`gemma4:e2b`** (Ollama stock, 1.8 GB resident) | — | **26.1** | — |
| **Ollama** `qwen2.5:3b` `Q4_K_M` (stock) | 3.0 | **23.5** | **7.8×** |
| Ollama `qwen2.5:3b` + flash-attn (f16 KV) | — | 23.5 | — |
| Ollama `qwen2.5:3b` + flash-attn + `q8_0` KV | — | 22.9 | — |
| vLLM `0.22` (bnb, `--enforce-eager`) | 5.7 | 7.0 | 1.2× |
| `models` image transformers (bnb-nf4) | 4.6 | 5.7 | 1.2× |

Ollama **config tuning does not add text speed**: `OLLAMA_FLASH_ATTENTION=1`
(23.5, unchanged) and `+OLLAMA_KV_CACHE_TYPE=q8_0` (22.9, slightly slower) only
cut attention/KV memory — a tiny slice of a batch-1 3B decode, which is bound on
reading the ~2 GB of weights per token. So stock Ollama is already at its
single-stream ceiling; those knobs are **memory savers, not speedups**. Going
faster needs a **smaller model** (`gemma4:e2b` = 26.1) or a **different engine**
(tuned vLLM = 36.3), not Ollama flags.

Two independent findings:

- **Power mode is the ~8× lever.** Ollama scales 3.0 → 23.5 with MAXN because
  llama.cpp's *fused* GGUF kernels are **memory-bandwidth bound**, so unlocking
  EMC clocks lifts it straight to **~40% of the ~56 tok/s ceiling** (4-bit 3B ÷
  102 GB/s) — *usable* chat, no code change. The earlier "Ollama kernels are
  unoptimized" guess was wrong; it was throttled.
- **Engine tuning is the second lever.** The bnb + `--enforce-eager` vLLM/
  transformers rows are **software-overhead-bound** (per-token dequant + Python
  dispatch, GPU mostly idle), so the clock unlock barely helps them (1.2×). Swap
  bnb → **AWQ-Marlin** and turn **CUDA graphs on** and vLLM becomes hardware-
  bound too: **36.3 tok/s, ~1.5× Ollama** (verified; AWQ-Marlin + CUDA graphs
  both run on Orin `sm_87`, no error). transformers stays slow — torch-2.13's
  Triton-eager routing (the `sm_87` note below) — and is a dead end for text.

**But speed is not the only axis on 8 GB.** vLLM reserves VRAM *statically* and
has a hard floor: at `--gpu-memory-utilization 0.45` (~3.3 GB) it still runs full
speed (**35.8 tok/s** with CUDA graphs), but **0.40 (~3.0 GB) and below OOM** —
weights + the CUDA-graph pool + minimum KV do not fit. You **cannot** trade that
graph memory back for footprint: `--enforce-eager` at the same util drops it to
**16.6 tok/s**, so graphs *are* the speed. **vLLM's fast floor is ~3.3 GB and it
cannot reach Ollama's ~2.1 GB.** That fed the earlier Ollama-for-text choice, but
plan 16 supersedes it: a source-built `sm_87` `llama-server` running the single
gemma4-E2B GGUF does text **and** vision on the GPU at ~2 GB resident (30 tok/s
text, sub-second vision — below), beating both Ollama's 26.1 and its broken
CPU-projector vision, and leaving SigLIP (~1.6 GB) + the nomic embedder (~0.3 GB)
room inside ~5.8 GB. So the **standing recommendation is MAXN_SUPER + one
gemma4-E2B on `llama-server`**; tuned vLLM (35.8 tok/s text-only) stays a
cloud-side option. Single-stream decode is bandwidth-bound, so 4-bit (fewer
bytes/token) is the *fast* choice and the 67 INT8 TOPS does not help chat latency;
power mode is the first lever, and the `sm_87` build is what unlocks GPU vision.

## GPU vision benchmark — via a source-built `sm_87` llama.cpp (verified 2026-08-16)

Stock Ollama runs the vision projector (CLIP/`mmproj`) on **CPU** (~190–211
s/image — the reason captioning went in-process). But a `llama.cpp` built
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` (inside the
`ghcr.io/nvidia-ai-iot/ollama:…cu130` image, which carries nvcc/cmake; the cu130
binary runs on the cu132 driver, no crash), run with the projector **on GPU**
(default offload) + `-ngl 99 --flash-attn on --jinja -c 4096`, measures — one
GGUF serving **both** text and vision:

| Model (`sm_87` llama.cpp, GPU, MAXN) | Text tok/s | Vision (image encode) |
|---|---|---|
| **`gemma4-E2B`** (Q4_K_M + f16 mmproj) | **30.0** | **0.57 s/img** |
| `Qwen2.5-VL-3B` (Q4_K_M + Q8 mmproj) | 23.5 | 13.8 s/img |

`gemma4-E2B` wins on **both** axes: faster text than Ollama (30 vs 26.1) and
far faster vision (0.57 s *encode* vs Qwen's 13.8 s — Qwen emits ~2048 image
tokens, gemma ~256). End-to-end per caption is ~5–6 s for gemma vs ~20 s for Qwen
(add generation). CPU projector (`--no-mmproj-offload`) is ~149 s for Qwen, so GPU
offload is ~10–370×.

Two gemma4 config requirements (verified 2026-08-16):
- **`--jinja`** — its chat template aborts otherwise (the earlier "empty caption").
- **`--chat-template-kwargs '{"enable_thinking":false}'`** — without it gemma4
  dumps a verbose chain-of-thought instead of a caption; `--reasoning-budget 0`
  is *not* enough. With thinking off + a captioning system prompt, gemma4-E2B
  **reads in-image text and fine detail at Qwen2.5-VL quality** (bake-off on a bus
  photo: both read the route/EMT/"cero emisiones" text; gemma also caught the
  wrought-iron balconies) — so gemma's speed does **not** cost quality.

This is the basis of plan `16` (single Gemma on a `llama-server`, text + vision on
GPU, replacing qwen-text + the in-process VLM, and dropping Ollama on jetson).

> **NOTE — Orin `sm_87` cubins.** The generic cu132 torch wheels **do not** ship
> `sm_87` cubins, so torch kernels JIT from PTX or fall back — historically the
> cause of slow in-process decode. This no longer affects inference (gemma text +
> vision runs on the `sm_87`-native `llama-server`, not torch); it can still slow
> **SigLIP** in the `models` service. If SigLIP throughput matters, pin a torch
> build that carries `sm_87` cubins or use an NVIDIA Jetson wheel index.
