#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure ~/.cargo/bin is in PATH for community bindgen tools
if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "=== Generating canonical C header with cbindgen ==="
cargo run --features cbindgen-cli --bin generate-c-headers

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

if command -v uniffi-bindgen-dart >/dev/null 2>&1; then
    echo "=== Generating Dart bindings ==="
    mkdir -p packages/dart/lib/src/generated
    uniffi-bindgen-dart generate "$HOST_LIB" --out-dir packages/dart/lib/src/generated
else
    echo "Skipping Dart generation: uniffi-bindgen-dart not found in PATH"
fi

if command -v uniffi-bindgen-cs >/dev/null 2>&1; then
    echo "=== Generating C# (.NET) bindings ==="
    mkdir -p packages/dotnet/generated
    uniffi-bindgen-cs --library "$HOST_LIB" --config uniffi.toml --out-dir packages/dotnet/generated
else
    echo "Skipping C# generation: uniffi-bindgen-cs not found in PATH"
fi

if command -v uniffi-bindgen-go >/dev/null 2>&1; then
    echo "=== Generating Go bindings ==="
    mkdir -p packages/go/uniffi
    uniffi-bindgen-go "$HOST_LIB" --out-dir packages/go/uniffi
else
    echo "Skipping Go generation: uniffi-bindgen-go not found in PATH"
fi

echo "=== Bindings generation complete ==="
