#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
WASM_DIR="$REPO_ROOT/packages/web/src/wasm"

mkdir -p "$WASM_DIR" "$DOCS_DIR/public/wasm"

if [ -f "$WASM_DIR/rusto_rten_wasm.js" ] && [ -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" ]; then
  echo "✓ Local WASM runtime already present in $WASM_DIR"
  cp -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/" || true
  exit 0
fi

echo "Fetching prebuilt WASM runtime for documentation..."
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if gh release download --pattern 'rusto-wasm.tar.gz' --dir "$TMP_DIR" 2>/dev/null; then
  tar -xzf "$TMP_DIR/rusto-wasm.tar.gz" -C "$WASM_DIR"
elif curl -sSLf "https://github.com/byrizki/rusto-rs/releases/latest/download/rusto-wasm.tar.gz" -o "$TMP_DIR/rusto-wasm.tar.gz"; then
  tar -xzf "$TMP_DIR/rusto-wasm.tar.gz" -C "$WASM_DIR"
elif [ -x "$(command -v wasm-pack)" ]; then
  echo "Falling back to local wasm-pack build..."
  (cd "$REPO_ROOT" && wasm-pack build crates/rten-wasm --target web --out-dir "$WASM_DIR")
else
  echo "Error: Failed to obtain WASM runtime. Neither release assets nor wasm-pack available." >&2
  exit 1
fi

if [ -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" ]; then
  cp -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/"
fi

echo "✓ WASM runtime successfully prepared in $WASM_DIR and $DOCS_DIR/public/wasm"
