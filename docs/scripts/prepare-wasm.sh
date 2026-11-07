#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
WASM_DIR="$REPO_ROOT/packages/web/src/wasm"

mkdir -p "$WASM_DIR" "$DOCS_DIR/public/wasm"

# 1. If wasm-pack is installed, compile from source to ensure latest build
if command -v wasm-pack >/dev/null 2>&1; then
  echo "Building pluggable WASM inference runtimes from source using wasm-pack..."

  echo "  1/3 Building rusto-nn (default pure-Rust inference engine)..."
  (cd "$REPO_ROOT" && wasm-pack build crates/rusto-wasm --target web --release --out-name rusto_nn --out-dir "$WASM_DIR")

  echo "  2/3 Building rusto-rten (RTen inference engine)..."
  (cd "$REPO_ROOT" && wasm-pack build crates/rusto-wasm --target web --release --out-name rusto_rten --out-dir "$WASM_DIR" -- --no-default-features --features rusto-rten)

  echo "  3/3 Building rusto-webgpu (WebGPU hardware accelerator engine)..."
  (cd "$REPO_ROOT" && wasm-pack build crates/rusto-wasm --target web --release --out-name rusto_webgpu --out-dir "$WASM_DIR" -- --no-default-features --features rusto-webgpu)

  rm -f "$WASM_DIR/.gitignore"

  # Copy all WASM binaries to docs/public/wasm
  cp -f "$WASM_DIR"/*.wasm "$DOCS_DIR/public/wasm/"
  echo "✓ WASM runtimes (rusto-nn, rusto-rten, rusto-webgpu) successfully built in $WASM_DIR and $DOCS_DIR/public/wasm"
  exit 0
fi

# 2. Fallback: if local WASM runtimes already present, copy to public
if [ -f "$WASM_DIR/rusto_nn_bg.wasm" ]; then
  echo "✓ Local WASM runtimes already present in $WASM_DIR"
  cp -f "$WASM_DIR"/*.wasm "$DOCS_DIR/public/wasm/" || true
  exit 0
fi

echo "Error: wasm-pack is not installed and no local WASM runtime was found." >&2
echo "Please install wasm-pack ('curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh') to build from source." >&2
exit 1
