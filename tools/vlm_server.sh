#!/usr/bin/env bash
# Starts the local vision-language reasoner (llama.cpp) for the "Explain" button.
#
# Sizing, for this machine (6 physical cores, AVX-512 + VNNI, no usable GPU):
#   -t 6        one thread per physical core. Hyper-threads make llama.cpp slower, not faster, and leaving
#               the other 6 free means the detector keeps running if a camera is analysing at the same time.
#   -c 8192     enough for two 768-px photographs plus the prompt and the answer. A larger context would
#               only reserve more KV cache memory for nothing.
#   --image-min-tokens 1024
#               Qwen-VL degrades badly below this: the model's own loader warns about it. Costs a few
#               seconds per photograph and is the difference between the model reading a scene and guessing.
#   --load-mode mmap+mlock
#               maps the weights and keeps them resident. This box has 15 GB of swap; paging a model out
#               mid-answer turns 40 seconds into minutes.
# Resident size is about 3.5-4 GB, against ~14 GB free here. It cannot exhaust memory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/vendor/llama.cpp/llama-server"
VLM="$ROOT/models/vlm"
MODEL="$VLM/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ="$VLM/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

PORT="${VLM_PORT:-8081}"
THREADS="${VLM_THREADS:-$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)}"
CTX="${VLM_CTX:-8192}"

if [ ! -x "$BIN" ] || [ ! -f "$MODEL" ] || [ ! -f "$MMPROJ" ]; then
    echo "The local reasoner is not installed yet. Run:  ./tools/setup_vlm.sh" >&2
    exit 1
fi

echo "llama-server on port $PORT  ($THREADS threads, ${CTX}-token context)"
echo "Leave this running and use the Explain button in the dashboard. Ctrl-C to stop."
exec "$BIN" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --alias qwen2.5-vl-3b \
    --host 127.0.0.1 --port "$PORT" \
    --threads "$THREADS" \
    --ctx-size "$CTX" \
    --batch-size 256 \
    --image-min-tokens 1024 \
    --temp 0.2 \
    --load-mode mmap+mlock \
    --no-webui \
    "$@"
