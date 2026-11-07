#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
TARGET_DIR="${1:-$DOCS_DIR/public/models}"

HF_BASE="${HF_BASE:-https://huggingface.co/byrizki/rusto-models/resolve/main}"

mkdir -p "$TARGET_DIR"

# Only bundle client/web-appropriate models for live demo (<25 MB each, avoids GitHub Pages 100 MB limit)
declare -A MODEL_FILES=(
  ["indraocr"]="indra_det.mnn indra_rec.mnn indra_meta.json"
  ["ppocrv6-tiny"]="ppocrv6_tiny_det.mnn ppocrv6_tiny_rec.mnn ppocrv6_dict.txt"
  ["ppocrv6-tiny-int8"]="ppocrv6_tiny_int8_det.onnx ppocrv6_tiny_int8_rec.onnx ppocrv6_dict.txt"
  ["ppocrv6-small"]="ppocrv6_small_det.onnx ppocrv6_small_rec.onnx ppocrv6_dict.txt"
  ["ppocrv6-small-int8"]="ppocrv6_small_int8_det.onnx ppocrv6_small_int8_rec.onnx ppocrv6_dict.txt"
  ["ppocrv5-mobile"]="ppocrv5_mobile_det.onnx ppocrv5_mobile_rec.onnx ppocrv5_dict.txt"
  ["ppocrv4-mobile"]="ppocrv4_mobile_det.onnx ppocrv4_mobile_rec.onnx ppocrv4_dict.txt"
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

  det_ext="${det_file##*.}"
  rec_ext="${rec_file##*.}"

  if [ "$model" = "indraocr" ]; then
    download_file "$TARGET_DIR/$model/det.$det_ext" "$model" "det.$det_ext" "$det_file" "$REPO_ROOT/models/indra/det.$det_ext"
    download_file "$TARGET_DIR/$model/rec.$rec_ext" "$model" "rec.$rec_ext" "$rec_file" "$REPO_ROOT/models/indra/rec.$rec_ext"
    download_file "$TARGET_DIR/$model/meta.json" "$model" "meta.json" "$dict_file" "$REPO_ROOT/models/indra/meta.json"
    link_flat "$TARGET_DIR/$model/det.$det_ext" "$det_file"
    link_flat "$TARGET_DIR/$model/rec.$rec_ext" "$rec_file"
    link_flat "$TARGET_DIR/$model/meta.json" "$dict_file"
  else
    download_file "$TARGET_DIR/$model/det.$det_ext" "$model" "det.$det_ext" "$det_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/det.$det_ext"
    download_file "$TARGET_DIR/$model/rec.$rec_ext" "$model" "rec.$rec_ext" "$rec_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/rec.$rec_ext"
    download_file "$TARGET_DIR/$model/dict.txt" "$model" "dict.txt" "$dict_file" "$REPO_ROOT/models/PPOCR_${model//-/_}/dict.txt"
    link_flat "$TARGET_DIR/$model/det.$det_ext" "$det_file"
    link_flat "$TARGET_DIR/$model/rec.$rec_ext" "$rec_file"
    link_flat "$TARGET_DIR/$model/dict.txt" "$dict_file"
  fi
done

# Also download layout and auxiliary modules for live demo
mkdir -p "$TARGET_DIR/modules"
download_file "$TARGET_DIR/modules/layout_cdla.mnn" "modules" "layout_cdla.mnn" "layout_cdla.mnn" "$REPO_ROOT/models/layout_cdla.mnn"
download_file "$TARGET_DIR/modules/layout_publaynet.mnn" "modules" "layout_publaynet.mnn" "layout_publaynet.mnn" "$REPO_ROOT/models/layout_publaynet.mnn"
download_file "$TARGET_DIR/modules/orient.mnn" "modules" "orient.mnn" "orient.mnn" "$REPO_ROOT/models/orient.mnn"
download_file "$TARGET_DIR/modules/cls.mnn" "modules" "cls.mnn" "cls.mnn" "$REPO_ROOT/models/cls.mnn"
download_file "$TARGET_DIR/modules/uvdoc.mnn" "modules" "uvdoc.mnn" "uvdoc.mnn" "$REPO_ROOT/models/uvdoc.mnn"

# Layout / reading order model for Indra OCR
download_file "$TARGET_DIR/indraocr/order.rten" "indraocr" "order.rten" "indra_order.rten" "$REPO_ROOT/models/indra/order.rten"
link_flat "$TARGET_DIR/indraocr/order.rten" "indra_order.rten"
download_file "$TARGET_DIR/indraocr/order.onnx" "indraocr" "order.onnx" "indra_order.onnx" "$REPO_ROOT/models/indra/order.onnx"
link_flat "$TARGET_DIR/indraocr/order.onnx" "indra_order.onnx"

echo "=== Syncing WebAssembly runtime binary ==="
mkdir -p "$DOCS_DIR/public/wasm"
if [ -f "$REPO_ROOT/packages/web/src/wasm/rusto_nn_bg.wasm" ]; then
  cp -f "$REPO_ROOT/packages/web/src/wasm"/*.wasm "$DOCS_DIR/public/wasm/"
  echo "✓ Synced WASM runtimes to public/wasm/"
fi

echo "✅ Models ready in $TARGET_DIR"
