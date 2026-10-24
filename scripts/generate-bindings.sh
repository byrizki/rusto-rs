#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Building host library with UniFFI & C FFI ==="
cargo build --features "ffi,uniffi-cli" --lib

HOST_LIB="$(find target/debug -maxdepth 1 \( -name 'librusto.so' -o -name 'librusto.dylib' -o -name 'rusto.dll' \) | head -n 1)"
if [[ -z "$HOST_LIB" ]]; then
    echo "Error: Host library not found in target/debug" >&2
    exit 1
fi

echo "Found host library: $HOST_LIB"

echo "=== Generating Kotlin bindings ==="
mkdir -p packages/android/src/main/kotlin/com/byrizki/rusto/generated
cargo run --features uniffi-cli --bin uniffi-bindgen generate \
    --library "$HOST_LIB" \
    --language kotlin \
    --out-dir packages/android/src/main/kotlin/com/byrizki/rusto/generated

echo "=== Generating Swift bindings ==="
mkdir -p packages/ios/src/generated
cargo run --features uniffi-cli --bin uniffi-bindgen generate \
    --library "$HOST_LIB" \
    --language swift \
    --out-dir packages/ios/src/generated

echo "=== Generating Python bindings ==="
mkdir -p packages/python/rusto/generated
cargo run --features uniffi-cli --bin uniffi-bindgen generate \
    --library "$HOST_LIB" \
    --language python \
    --out-dir packages/python/rusto/generated

echo "=== Bindings generation complete ==="
