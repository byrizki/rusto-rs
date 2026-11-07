#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
TARGET_DIR="${1:-$DOCS_DIR/public/models}"

HF_BASE="${HF_BASE:-https://huggingface.co/byrizki/rten-models/resolve/main}"

mkdir -p "$TARGET_DIR"

# Only bundle client/web-appropriate models for live demo (<25 MB each, avoids GitHub Pages 100 MB limit)
declare -A MODEL_FILES=(
  ["indraocr"]="indra_det.rten indra_rec.rten indra_meta.json"
  ["ppocrv6-tiny"]="ppocrv6_tiny_det.rten ppocrv6_tiny_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-tiny-int8"]="ppocrv6_tiny_int8_det.rten ppocrv6_tiny_int8_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-small"]="ppocrv6_small_det.rten ppocrv6_small_rec.rten ppocrv6_dict.txt"
  ["ppocrv6-small-int8"]="ppocrv6_small_int8_det.rten ppocrv6_small_int8_rec.rten ppocrv6_dict.txt"
  ["ppocrv5-mobile"]="ppocrv5_mobile_det.rten ppocrv5_mobile_rec.rten ppocrv5_dict.txt"
  ["ppocrv4-mobile"]="ppocrv4_mobile_det.rten ppocrv4_mobile_rec.rten ppocrv4_dict.txt"
)

echo "=== Indexing and Downloading Models from Hugging Face Hub ($HF_BASE) ==="

download_file() {
  local target="$1"
  local model="$2"
  local file_name="$3"
  local release_file="$4"
  local local_cand="$5"

  if [ -s "$target" ]; then
    return 0
  fi

  if [ -n "$local_cand" ] && [ -s "$local_cand" ]; then
    echo "  ✓ Copying from local candidate: $local_cand"
    cp "$local_cand" "$target"
    return 0
  fi

  echo "  ⬇️ Downloading $model/$file_name from Hugging Face..."
  if curl -sSLf "$HF_BASE/$model/$file_name" -o "$target" 2>/dev/null; then
    return 0
  fi
  if curl -sSLf "$HF_BASE/$model/$release_file" -o "$target" 2>/dev/null; then
    return 0
  fi
  echo "  ⚠️ Model download failed from Hugging Face for $model/$file_name ($release_file)"
}

for model in "${!MODEL_FILES[@]}"; do
  mkdir -p "$TARGET_DIR/$model"
  read -r det_file rec_file dict_file <<< "${MODEL_FILES[$model]}"

  link_flat() {
    local src="$1"
    local alias="$2"
    if [ -f "$src" ]; then
      ln -f "$src" "$TARGET_DIR/$alias" 2>/dev/null || cp -f "$src" "$TARGET_DIR/$alias" 2>/dev/null || true
    fi
  }

  if [ "$model" = "indraocr" ]; then
    download_file "$TARGET_DIR/$model/det.rten" "$model" "det.rten" "$det_file" "$REPO_ROOT/models/indra/det.rten"
    download_file "$TARGET_DIR/$model/rec.rten" "$model" "rec.rten" "$rec_file" "$REPO_ROOT/models/indra/rec.rten"
    download_file "$TARGET_DIR/$model/meta.json" "$model" "meta.json" "$dict_file" "$REPO_ROOT/models/indra/meta.json"
    link_flat "$TARGET_DIR/$model/det.rten" "$det_file"
    link_flat "$TARGET_DIR/$model/rec.rten" "$rec_file"
    link_flat "$TARGET_DIR/$model/meta.json" "$dict_file"
  else
    download_file "$TARGET_DIR/$model/det.rten" "$model" "det.rten" "$det_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/det.rten"
    download_file "$TARGET_DIR/$model/rec.rten" "$model" "rec.rten" "$rec_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/rec.rten"
    download_file "$TARGET_DIR/$model/dict.txt" "$model" "dict.txt" "$dict_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/dict.txt"
    link_flat "$TARGET_DIR/$model/det.rten" "$det_file"
    link_flat "$TARGET_DIR/$model/rec.rten" "$rec_file"
    link_flat "$TARGET_DIR/$model/dict.txt" "$dict_file"
  fi
done

echo "=== Syncing WebAssembly runtime binary ==="
mkdir -p "$DOCS_DIR/public/wasm"
if [ -f "$REPO_ROOT/packages/web/src/wasm/rusto_nn_bg.wasm" ]; then
  cp -f "$REPO_ROOT/packages/web/src/wasm"/*.wasm "$DOCS_DIR/public/wasm/"
  echo "✓ Synced WASM runtimes to public/wasm/"
fi

echo "✅ Models ready in $TARGET_DIR"
