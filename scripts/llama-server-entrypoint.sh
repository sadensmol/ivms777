#!/bin/sh
# Fetch the gemma4-E2B GGUFs (once, idempotent) into $LLAMA_MODEL_DIR, then hand
# off to llama-server with whatever flags the compose `command` passed (plan 16).
#
# `LLAMA_BIN` selects the server binary: on jetson it points at the PREBUILT
# binary mounted from the `llamacpp` volume (built once, Aug 2026 — never
# recompiled); on mac it is just `llama-server` on PATH. The model files + their
# Hugging Face URLs are env-overridable, and the compose `command` points at the
# same /models/<file> paths.
set -eu

LLAMA_BIN="${LLAMA_BIN:-llama-server}"
MODEL_DIR="${LLAMA_MODEL_DIR:-/models}"
GGUF="${LLAMA_GGUF:-gemma-4-E2B-it-Q4_K_M.gguf}"
# Q8_0 projector (531 MB), NOT F16 (985 MB): everything stays on the GPU (design
# §3.1), and the F16 projector's CLIP buffer is what OOM-aborted llama-server on the
# 8 GB board. It ships in ggml-org's repo, not unsloth's — hence a separate repo var.
MMPROJ="${LLAMA_MMPROJ:-mmproj-gemma-4-E2B-it-Q8_0.gguf}"
REPO="${LLAMA_GGUF_REPO:-unsloth/gemma-4-E2B-it-GGUF}"
MMPROJ_REPO="${LLAMA_MMPROJ_REPO:-ggml-org/gemma-4-E2B-it-GGUF}"
GGUF_URL="${LLAMA_GGUF_URL:-https://huggingface.co/${REPO}/resolve/main/${GGUF}}"
MMPROJ_URL="${LLAMA_MMPROJ_URL:-https://huggingface.co/${MMPROJ_REPO}/resolve/main/${MMPROJ}}"

mkdir -p "$MODEL_DIR"
fetch() {  # fetch <url> <dest> — reuse if present; curl or wget, whichever exists
  [ -s "$2" ] && { echo "llama-server: reusing $(basename "$2")"; return; }
  echo "llama-server: fetching $(basename "$2") …"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$2.part" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget --progress=dot:giga -O "$2.part" "$1"
  else
    echo "llama-server: neither curl nor wget available to fetch $1" >&2; exit 1
  fi
  mv "$2.part" "$2"
}
fetch "$GGUF_URL" "$MODEL_DIR/$GGUF"
fetch "$MMPROJ_URL" "$MODEL_DIR/$MMPROJ"

echo "llama-server: starting ($LLAMA_BIN) — $*"
exec "$LLAMA_BIN" "$@"
