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
# Resident size is about 3.5-4 GB for the 3B (see docs/VLM.md for the 2B and 8B measurements).
#
# Tiers (the two-tier cascade, surveillance/vlm_cascade.py):
#   --tier screen   Qwen3-VL-2B,  port 8081, meant to be always on (every case is screened by it)
#   --tier reason   Qwen3-VL-8B,  port 8082, normally started on demand by the dashboard server and
#                   stopped after an idle timeout; start it by hand only to measure or debug
#   (none)          the single Qwen2.5-VL-3B on 8081: the pre-cascade setup, kept as the fallback
# Override with VLM_PORT, VLM_THREADS, VLM_CTX. Remaining arguments go to llama-server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/vendor/llama.cpp/llama-server"
VLM="$ROOT/models/vlm"
TIER="single"
if [ "${1:-}" = "--tier" ]; then TIER="${2:-}"; shift 2 || true; fi
case "$TIER" in
    screen) MODEL="$VLM/Qwen3VL-2B-Instruct-Q4_K_M.gguf"; MMPROJ="$VLM/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
            ALIAS="qwen3-vl-2b"; DEF_PORT=8081 ;;
    reason) MODEL="$VLM/Qwen3VL-8B-Instruct-Q4_K_M.gguf"; MMPROJ="$VLM/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
            ALIAS="qwen3-vl-8b"; DEF_PORT=8082 ;;
    single) MODEL="$VLM/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"; MMPROJ="$VLM/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
            ALIAS="qwen2.5-vl-3b"; DEF_PORT=8081 ;;
    *) echo "usage: $0 [--tier screen|reason] [llama-server args]" >&2; exit 2 ;;
esac

PORT="${VLM_PORT:-$DEF_PORT}"
THREADS="${VLM_THREADS:-$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)}"
CTX="${VLM_CTX:-8192}"

if [ ! -x "$BIN" ] || [ ! -f "$MODEL" ] || [ ! -f "$MMPROJ" ]; then
    echo "The local reasoner is not installed yet. Run:  ./tools/setup_vlm.sh" >&2
    exit 1
fi

echo "llama-server [$TIER: $ALIAS] on port $PORT  ($THREADS threads, ${CTX}-token context)"
echo "Leave this running and use the Explain button in the dashboard. Ctrl-C to stop."
exec "$BIN" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --alias "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" \
    --threads "$THREADS" \
    --ctx-size "$CTX" \
    --batch-size 256 \
    --image-min-tokens 1024 \
    --temp 0.2 \
    --load-mode mmap+mlock \
    --no-webui \
    "$@"
