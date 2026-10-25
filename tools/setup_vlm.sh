#!/usr/bin/env bash
# Downloads the local vision-language reasoner: the llama.cpp server binary and a quantised Qwen2.5-VL.
#
# Everything lands under the project (models/vlm, vendor/llama.cpp), never in the system or the home
# partition: on this machine / has little room left while the project's own partition has plenty.
# Safe to re-run: finished files are skipped and interrupted downloads resume where they stopped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLM_DIR="$ROOT/models/vlm"
VENDOR="$ROOT/vendor/llama.cpp"

LLAMA_BUILD="${LLAMA_BUILD:-b11856}"
LLAMA_TGZ="llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"   # CPU build: AVX-512/VNNI is picked at runtime
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${LLAMA_TGZ}"

REPO="https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main"
WEIGHTS="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"            # 1.8 GB  language model
MMPROJ="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"         # 1.3 GB  vision encoder

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

fetch() {  # url dest  -- resumable, skips a file that is already complete
    local url="$1" dest="$2" want have
    want=$(curl -sSLI -m 30 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9' || true)
    have=$( [ -f "$dest" ] && stat -c %s "$dest" || echo 0 )
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
        echo "  already complete: $(basename "$dest") ($(numfmt --to=iec "$have"))"
        return
    fi
    echo "  downloading $(basename "$dest") ($(numfmt --to=iec "${want:-0}"))"
    # A progress bar redirected to a log file is thousands of useless lines, so only draw one on a terminal.
    local meter=(--no-progress-meter); [ -t 1 ] && meter=(--progress-bar)
    curl -fL --retry 10 --retry-all-errors --retry-delay 5 -C - "${meter[@]}" -o "$dest" "$url"
    have=$(stat -c %s "$dest")
    if [ -n "$want" ] && [ "$have" != "$want" ]; then
        echo "  !! $(basename "$dest") is $have bytes, expected $want. Re-run to resume." >&2
        exit 1
    fi
}

mkdir -p "$VLM_DIR" "$VENDOR"

say "1/3  llama.cpp server ($LLAMA_BUILD, CPU build)"
if [ -x "$VENDOR/llama-server" ]; then
    echo "  already installed: $VENDOR/llama-server"
else
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    fetch "$LLAMA_URL" "$tmp/$LLAMA_TGZ"
    tar -xzf "$tmp/$LLAMA_TGZ" -C "$tmp"
    # The archive layout has changed between releases: find the binary wherever it is.
    src="$(find "$tmp" -type f -name llama-server | head -1)"
    [ -n "$src" ] || { echo "  !! llama-server not found inside $LLAMA_TGZ" >&2; exit 1; }
    cp -a "$(dirname "$src")"/. "$VENDOR"/
    chmod +x "$VENDOR"/llama-* 2>/dev/null || true
    echo "  installed into $VENDOR"
fi

say "2/3  vision-language model (~3.1 GB, resumable)"
fetch "$REPO/$WEIGHTS" "$VLM_DIR/$WEIGHTS"
fetch "$REPO/$MMPROJ" "$VLM_DIR/$MMPROJ"

say "3/3  check"
"$VENDOR/llama-server" --version 2>&1 | head -3 || true
ls -lh "$VLM_DIR"/*.gguf | awk '{print "  "$9"  "$5}'

say "Done. Start the reasoner with:  ./tools/vlm_server.sh"
