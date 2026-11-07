# rusto-mnn 🏎️

[![Crates.io](https://img.shields.io/crates/v/rusto-mnn.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-mnn)
[![docs.rs](https://img.shields.io/docsrs/rusto-mnn?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-mnn)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Native C++ FFI bindings to the official [Alibaba MNN](https://github.com/alibaba/MNN) deep learning framework for [RustO!](https://github.com/byrizki/rusto-rs).

`rusto-mnn` integrates the upstream C++ MNN runtime into RustO!'s `InferenceEngine` interface, serving as the high-precision ground truth for latency, throughput, and accuracy comparisons against RustO's pure-Rust `rusto-nn` engine.

---

## 🚀 Features

- **Official MNN C++ Runtime** — Compiles and links upstream Alibaba MNN static engine via `cc` in `build.rs`.
- **InferenceEngine Adapter** — Implements `rusto_core::engine::InferenceEngine` and `InferenceSession` for `.mnn` models.
- **Benchmark & Verification Baseline** — Validates output tensors, numerical precision, and inference latency against pure Rust implementations.

---

## 📦 Requirements

- A modern C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.13+ (for compiling upstream MNN sources)

---

## 📦 Installation

Add `rusto-mnn` to your `Cargo.toml`:

```toml
[dependencies]
rusto-mnn = "0.4"
rusto-core = "0.4"
```

---

## 💡 Usage

```rust
use rusto_core::engine::InferenceEngine;
use rusto_mnn::MnnCppEngine;

// Create MNN C++ engine instance
let engine = MnnCppEngine::new();
let session = engine.create_session("model.mnn")?;

// Or register globally with RustO
rusto_mnn::register();
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

