# rusto-nn 🧠

[![Crates.io](https://img.shields.io/crates/v/rusto-nn.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-nn)
[![docs.rs](https://img.shields.io/docsrs/rusto-nn?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-nn)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

A lightweight, zero-bloat, **100% pure-Rust** deep learning inference engine for Alibaba MNN (`.mnn`), RTen (`.rten`), and ONNX (`.onnx`) models.

Developed as the core inference engine of [RustO!](https://github.com/byrizki/rusto-rs), `rusto-nn` delivers fast, memory-safe neural network execution without C++ compilers, external BLAS packages, or platform-specific shared libraries.

---

## 🚀 Features

- **Multi-Format Model Loader** — Seamlessly auto-detects and executes models in:
  - Alibaba MNN format (`.mnn` FlatBuffers Net)
  - RTen format (`.rten` flat binary)
  - ONNX format (`.onnx` protocol buffers)
- **Zero C++ Dependencies** — Pure Rust implementation compiling directly with `cargo build` on any platform.
- **Hardware SIMD Vectorization** — Native AVX-512, AVX2 + FMA, ARM NEON, and WASM SIMD128 kernels for high-throughput convolution and matrix multiplication.
- **Pluggable Hardware Acceleration** — Optional pluggable backends for WebGPU ([`rusto-nn-webgpu`](../rusto-nn-webgpu)) and NVIDIA CUDA ([`rusto-nn-cuda`](../rusto-nn-cuda)).
- **Memory Optimized** — Reusable thread-local MatMul scratch buffers, zero-copy tensor slicing, and optional memory-mapped weight files via `memmap2`.
- **WASM Ready** — Runs directly in modern browsers and Web Workers with `wasm32-unknown-unknown` support.

---

## 📦 Installation

Add `rusto-nn` to your `Cargo.toml`:

```toml
[dependencies]
rusto-nn = "0.4"
```

### Feature Flags

| Feature | Default | Description |
|---|---|---|
| `std` | **Yes** | Standard library support |
| `rayon` | **Yes** | Multithreaded parallel execution for CPU operators |
| `rten` | **Yes** | Support for loading `.rten` model format files |
| `onnx` | **Yes** | Support for loading `.onnx` model format files |

---

## 💡 Quick Start

```rust
use rusto_nn::{Model, Session};
use std::sync::Arc;

// Load an MNN, ONNX, or RTen model from file or bytes
let model = Arc::new(Model::from_file("model.mnn")?);

// Create an execution session
let mut session = Session::from_arc(model)?;

// Prepare input tensor: [batch, channels, height, width]
let input_data = vec![0.0f32; 1 * 3 * 224 * 224];
session.set_input(0, &[1, 3, 224, 224], &input_data)?;

// Run inference
session.run()?;

// Retrieve output tensor
let output = session.get_output(0)?;
println!("Output shape: {:?}", output.shape());
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

