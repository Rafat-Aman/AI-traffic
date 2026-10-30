#!/usr/bin/env bash
# Downloads the local vision-language models: the llama.cpp server binary, the two-tier cascade models
# (Qwen3-VL-2B screens every case, Qwen3-VL-8B reasons on the escalated ones; both Q4_K_M + F16 mmproj) and the
# older single-model Qwen2.5-VL-3B (kept as the fallback).
#
#   ./tools/setup_vlm.sh              everything
#   ./tools/setup_vlm.sh cascade      only the 2B + 8B   (also: 3b, binary)
#
# Each Qwen3-VL file is checked against the SHA-256 that Hugging Face publishes for the LFS object.
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

QWEN3="https://huggingface.co/Qwen"
W2="Qwen3VL-2B-Instruct-Q4_K_M.gguf";  H_W2="089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
M2="mmproj-Qwen3VL-2B-Instruct-F16.gguf"; H_M2="c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5"
W8="Qwen3VL-8B-Instruct-Q4_K_M.gguf";  H_W8="67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"
M8="mmproj-Qwen3VL-8B-Instruct-F16.gguf"; H_M8="ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"
WHAT="${1:-all}"

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

verify() {  # file sha256 -- a size match is not proof; a hash is. Remembered in <file>.sha256ok
    local f="$1" want="$2" ok="$1.sha256ok"
    if [ -f "$ok" ] && [ "$(cat "$ok")" = "$want" ] && [ "$ok" -nt "$f" ]; then
        echo "  sha256 ok (cached): $(basename "$f")"; return
    fi
    echo "  checking sha256 of $(basename "$f") ..."
    local got; got=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "  !! sha256 mismatch for $(basename "$f")" >&2
        echo "     expected $want" >&2; echo "     got      $got" >&2
        echo "     delete the file and re-run." >&2; exit 1
    fi
    echo "$want" > "$ok"; echo "  sha256 ok: $(basename "$f")"
}

mkdir -p "$VLM_DIR" "$VENDOR"

want() { [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }

if want binary; then
say "1/4  llama.cpp server ($LLAMA_BUILD, CPU build)"
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
fi

if want cascade; then
say "2/4  screening model Qwen3-VL-2B (1.1 GB + 0.8 GB, resumable)"
fetch "$QWEN3/Qwen3-VL-2B-Instruct-GGUF/resolve/main/$W2" "$VLM_DIR/$W2"; verify "$VLM_DIR/$W2" "$H_W2"
fetch "$QWEN3/Qwen3-VL-2B-Instruct-GGUF/resolve/main/$M2" "$VLM_DIR/$M2"; verify "$VLM_DIR/$M2" "$H_M2"
say "3/4  reasoning model Qwen3-VL-8B (5.0 GB + 1.2 GB, resumable)"
fetch "$QWEN3/Qwen3-VL-8B-Instruct-GGUF/resolve/main/$W8" "$VLM_DIR/$W8"; verify "$VLM_DIR/$W8" "$H_W8"
fetch "$QWEN3/Qwen3-VL-8B-Instruct-GGUF/resolve/main/$M8" "$VLM_DIR/$M8"; verify "$VLM_DIR/$M8" "$H_M8"
fi

if want 3b; then
say "4/4  fallback single model Qwen2.5-VL-3B (~3.1 GB, resumable)"
fetch "$REPO/$WEIGHTS" "$VLM_DIR/$WEIGHTS"
fetch "$REPO/$MMPROJ" "$VLM_DIR/$MMPROJ"
fi

say "check"
"$VENDOR/llama-server" --version 2>&1 | head -3 || true
ls -lh "$VLM_DIR"/*.gguf | awk '{print "  "$9"  "$5}'

say "Done. Start the tiers with:  ./tools/vlm_server.sh --tier screen   (and --tier reason)"
