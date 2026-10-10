#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
TARGET_DIR="${1:-$DOCS_DIR/public/models}"

CDN_BASE="https://cdn.jsdelivr.net/npm"
MODELS=(
  "ppocrv6-tiny"
  "ppocrv6-tiny-int8"
  "ppocrv6-small"
  "ppocrv6-small-int8"
  "ppocrv6-medium"
  "ppocrv6-medium-int8"
  "ppocrv5-mobile"
  "ppocrv5-server"
  "ppocrv4-mobile"
  "ppocrv4-server"
)

echo "=== Downloading RTen models from jsDelivr CDN ==="
for model in "${MODELS[@]}"; do
  mkdir -p "$TARGET_DIR/$model"
  pkg="@rustors/model-$model"
  echo "Fetching $pkg from jsDelivr..."
  curl -sSL "$CDN_BASE/$pkg/models/det.rten" -o "$TARGET_DIR/$model/det.rten"
  curl -sSL "$CDN_BASE/$pkg/models/rec.rten" -o "$TARGET_DIR/$model/rec.rten"
  curl -sSL "$CDN_BASE/$pkg/models/dict.txt" -o "$TARGET_DIR/$model/dict.txt"
done

echo "=== Syncing WebAssembly runtime binary ==="
mkdir -p "$DOCS_DIR/public/wasm"
if [ -f "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" ]; then
  cp "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/"
  echo "✓ Synced rusto_rten_wasm_bg.wasm to public/wasm/"
fi

echo "✅ Models ready in $TARGET_DIR"
