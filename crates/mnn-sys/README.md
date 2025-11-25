# MNN-sys

Low-level FFI bindings for the MNN (Mobile Neural Network) library.

## Overview

This crate provides unsafe C bindings to MNN through a custom C wrapper layer. It is not intended to be used directly - use the `mnn` crate instead for safe, idiomatic Rust bindings.

## Building

### Requirements

- CMake 3.10 or higher
- C++11 compatible compiler (GCC 4.8+, Clang 3.3+, MSVC 2015+)
- Platform-specific dependencies:
  - **macOS/iOS**: Xcode with Metal SDK
  - **Linux**: Standard build tools
  - **Android**: NDK

### Build Process

The build script (`build.rs`) performs the following:

1. Compiles MNN from the vendored source using CMake
2. Compiles the C wrapper (`mnn_wrapper.cpp`)
3. Generates Rust bindings using bindgen
4. Links everything together

The build is configured for static linking by default to ensure the binaries are self-contained.

### Platform Configuration

The build script automatically enables platform-specific backends:

- **macOS/iOS**: Metal, CoreML
- **Linux**: OpenCL
- **Android**: OpenCL

## License

MNN is licensed under Apache 2.0. See the vendor/mnn directory for details.
