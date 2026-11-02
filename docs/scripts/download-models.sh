#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
TARGET_DIR="${1:-$DOCS_DIR/public/models}"

MODELS_TAG="${MODELS_TAG:-models-rten-v0.3.2}"
RELEASE_BASE="https://github.com/byrizki/rusto-rs/releases/download/$MODELS_TAG"

mkdir -p "$TARGET_DIR"

declare -A MODEL_FILES=(
  ["indraocr"]="indra_det.rten indra_rec.rten indra_meta.json"
  ["ppocrv6-tiny"]="ppocrv6_tiny_det.rten ppocrv6_tiny_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-tiny-int8"]="ppocrv6_tiny_int8_det.rten ppocrv6_tiny_int8_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-small"]="ppocrv6_small_det.rten ppocrv6_small_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-small-int8"]="ppocrv6_small_int8_det.rten ppocrv6_small_int8_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-medium"]="ppocrv6_medium_det.rten ppocrv6_medium_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-medium-int8"]="ppocrv6_medium_int8_det.rten ppocrv6_medium_int8_rec.rten ppocrv6_dict.txt"
  ["ppocrv5-mobile"]="ppocrv5_mobile_det.rten ppocrv5_mobile_rec.rten ppocrv5_dict.txt"
  ["ppocrv5-server"]="ppocrv5_server_det.rten ppocrv5_server_rec.rten ppocrv5_dict.txt"
  ["ppocrv4-mobile"]="ppocrv4_mobile_det.rten ppocrv4_mobile_rec.rten ppocrv4_dict.txt"
  ["ppocrv4-server"]="ppocrv4_server_det.rten ppocrv4_server_rec.rten ppocrv4_dict.txt"
)

echo "=== Indexing and Downloading Models from GitHub Release ($MODELS_TAG) ==="

download_file() {
  local target="$1"
  local release_file="$2"
  local local_cand="$3"

  if [ -s "$target" ]; then
    return 0
  fi

  if [ -n "$local_cand" ] && [ -s "$local_cand" ]; then
    echo "  ✓ Copying from local candidate: $local_cand"
    cp "$local_cand" "$target"
    return 0
  fi

  echo "  ⬇️ Downloading $release_file from GitHub Release..."
  if curl -sSLf "$RELEASE_BASE/$release_file" -o "$target" 2>/dev/null; then
    return 0
  fi
  echo "  ⚠️ GitHub Release download failed for $release_file"
}

for model in "${!MODEL_FILES[@]}"; do
  mkdir -p "$TARGET_DIR/$model"
  read -r det_file rec_file dict_file <<< "${MODEL_FILES[$model]}"

  if [ "$model" = "indraocr" ]; then
    download_file "$TARGET_DIR/$model/det.rten" "$det_file" "$REPO_ROOT/models/indra/det.rten"
    download_file "$TARGET_DIR/$model/rec.rten" "$rec_file" "$REPO_ROOT/models/indra/rec.rten"
    download_file "$TARGET_DIR/$model/meta.json" "$dict_file" "$REPO_ROOT/models/indra/meta.json"
  else
    download_file "$TARGET_DIR/$model/det.rten" "$det_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/det.rten"
    download_file "$TARGET_DIR/$model/rec.rten" "$rec_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/rec.rten"
    download_file "$TARGET_DIR/$model/dict.txt" "$dict_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/dict.txt"
  fi
done

echo "=== Syncing WebAssembly runtime binary ==="
mkdir -p "$DOCS_DIR/public/wasm"
if [ -f "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" ]; then
  cp -f "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/"
  echo "✓ Synced rusto_rten_wasm_bg.wasm to public/wasm/"
fi

echo "✅ Models ready in $TARGET_DIR"
