#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
WASM_DIR="$REPO_ROOT/packages/web/src/wasm"

mkdir -p "$WASM_DIR" "$DOCS_DIR/public/wasm"

# 1. If wasm-pack is installed, compile from source to ensure latest build
if command -v wasm-pack >/dev/null 2>&1; then
  echo "Building rten-wasm from source using wasm-pack..."
  (cd "$REPO_ROOT" && wasm-pack build crates/rten-wasm --target web --out-dir "$WASM_DIR")
  rm -f "$WASM_DIR/.gitignore"
  if [ -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" ]; then
    cp -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/"
  fi
  echo "✓ WASM runtime successfully built from source in $WASM_DIR and $DOCS_DIR/public/wasm"
  exit 0
fi

# 2. Fallback: if local WASM runtime already present, copy to public
if [ -f "$WASM_DIR/rusto_rten_wasm.js" ] && [ -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" ]; then
  echo "✓ Local WASM runtime already present in $WASM_DIR"
  cp -f "$WASM_DIR/rusto_rten_wasm_bg.wasm" "$DOCS_DIR/public/wasm/" || true
  exit 0
fi

echo "Error: wasm-pack is not installed and no local WASM runtime was found." >&2
echo "Please install wasm-pack ('curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh') to build from source." >&2
exit 1
