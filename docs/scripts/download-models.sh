#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
PUBLIC_MODELS="$DOCS_DIR/public/models"

mkdir -p "$PUBLIC_MODELS/ppocrv6-tiny"
mkdir -p "$PUBLIC_MODELS/ppocrv6-tiny-int8"
mkdir -p "$PUBLIC_MODELS/ppocrv6-small"
mkdir -p "$PUBLIC_MODELS/ppocrv6-small-int8"
mkdir -p "$PUBLIC_MODELS/ppocrv6-medium"
mkdir -p "$PUBLIC_MODELS/ppocrv6-medium-int8"
mkdir -p "$PUBLIC_MODELS/ppocrv5-mobile"
mkdir -p "$PUBLIC_MODELS/ppocrv5-server"
mkdir -p "$PUBLIC_MODELS/ppocrv4-mobile"
mkdir -p "$PUBLIC_MODELS/ppocrv4-server"

copy_or_download() {
  local target_id="$1"
  local local_dir="$2"
  local model_type="$3"
  local tier="$4"
  local target_dir="$PUBLIC_MODELS/$target_id"

  mkdir -p "$target_dir"
  if [ -d "$local_dir" ] && ([ -f "$local_dir/det.onnx" ] || [ -f "$local_dir/det.rten" ]); then
    echo "Copying local models from $local_dir to $target_dir/..."
    cp -r "$local_dir"/* "$target_dir/"
  else
    echo "Downloading $target_id via download_models.sh..."
    bash "$REPO_ROOT/scripts/download_models.sh" --model "$model_type" --tier "$tier" --output-dir "$target_dir"
  fi
}

echo "=== Preparing OCR models for docs dev preview ==="
copy_or_download "ppocrv6-tiny" "$REPO_ROOT/models/PPOCR_v6_tiny" "ppocrv6" "tiny"
copy_or_download "ppocrv6-small" "$REPO_ROOT/models/PPOCR_v6_small" "ppocrv6" "small"
copy_or_download "ppocrv6-medium" "$REPO_ROOT/models/PPOCR_v6_medium" "ppocrv6" "medium"
copy_or_download "ppocrv5-mobile" "$REPO_ROOT/models/PPOCR_v5_mobile" "ppocrv5" "mobile"
copy_or_download "ppocrv5-server" "$REPO_ROOT/models/PPOCR_v5_server" "ppocrv5" "server"
copy_or_download "ppocrv4-mobile" "$REPO_ROOT/models/PPOCR_v4_mobile" "ppocrv4" "mobile"
copy_or_download "ppocrv4-server" "$REPO_ROOT/models/PPOCR_v4_server" "ppocrv4" "server"

echo "=== Preparing PP-OCRv6 INT8 models for docs preview ==="
for tier in tiny small medium; do
  target_id="ppocrv6-${tier}-int8"
  target_dir="$PUBLIC_MODELS/$target_id"
  local_int8_dir="$REPO_ROOT/models/PPOCR_v6_${tier}_int8"
  mkdir -p "$target_dir"

  if [ -d "$local_int8_dir" ] && ([ -f "$local_int8_dir/det.rten" ] || [ -f "$local_int8_dir/det.onnx" ]); then
    echo "Copying local INT8 models from $local_int8_dir to $target_dir/..."
    cp -r "$local_int8_dir"/* "$target_dir/"
  elif [ -f "$PUBLIC_MODELS/ppocrv6-${tier}/det.onnx" ] || [ -f "$REPO_ROOT/models/PPOCR_v6_${tier}/det.onnx" ]; then
    echo "Quantizing PP-OCRv6 ${tier} to INT8..."
    PYTHON_CMD="python3"
    if [ -f "/tmp/rten-env/bin/python3" ]; then
      PYTHON_CMD="/tmp/rten-env/bin/python3"
    fi
    $PYTHON_CMD "$REPO_ROOT/scripts/quantize_ppocrv6_int8.py" --tier "$tier" --output-dir "$target_dir" || true
  fi
done

echo "=== Converting ONNX models to true .rten FlatBuffers format ==="
RTEN_CONVERT=""
if command -v rten-convert &>/dev/null; then
  RTEN_CONVERT="rten-convert"
elif [ -f "/tmp/rten-env/bin/rten-convert" ]; then
  RTEN_CONVERT="/tmp/rten-env/bin/rten-convert"
elif [ -f "$HOME/.local/bin/rten-convert" ]; then
  RTEN_CONVERT="$HOME/.local/bin/rten-convert"
else
  echo "Setting up rten-convert virtualenv in /tmp/rten-env..."
  python3 -m venv /tmp/rten-env
  /tmp/rten-env/bin/pip install -q rten-convert
  RTEN_CONVERT="/tmp/rten-env/bin/rten-convert"
fi

for mdir in "$PUBLIC_MODELS"/*; do
  [ -d "$mdir" ] || continue
  if [ -f "$mdir/det.onnx" ]; then
    echo "Converting $mdir/det.onnx -> $mdir/det.rten..."
    "$RTEN_CONVERT" "$mdir/det.onnx" "$mdir/det.rten"
    rm -f "$mdir/det.onnx"
  fi
  if [ -f "$mdir/rec.onnx" ]; then
    echo "Converting $mdir/rec.onnx -> $mdir/rec.rten..."
    "$RTEN_CONVERT" "$mdir/rec.onnx" "$mdir/rec.rten"
    rm -f "$mdir/rec.onnx"
  fi
  rm -f "$mdir"/*.onnx
done

echo "=== Syncing WebAssembly runtime binary ==="
mkdir -p "$DOCS_DIR/public/wasm"
if [ -f "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" ]; then
  cp "$REPO_ROOT/packages/web/src/wasm/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/"
  echo "✓ Synced rusto_rten_wasm_bg.wasm to public/wasm/"
fi

echo "✅ Docs preview models ready in $PUBLIC_MODELS:"
ls -lh "$PUBLIC_MODELS"/*
