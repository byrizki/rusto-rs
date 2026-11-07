# rusto-rten ⚡

[![Crates.io](https://img.shields.io/crates/v/rusto-rten.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-rten)
[![docs.rs](https://img.shields.io/docsrs/rusto-rten?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-rten)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Pluggable [RTen](https://github.com/robertknight/rten) inference engine backend for [RustO!](https://github.com/byrizki/rusto-rs).

`rusto-rten` adapts Robert Knight's lightweight, pure-Rust neural network engine into RustO!'s unified `InferenceEngine` interface, enabling native execution of `.rten` and `.onnx` model graphs.

---

## 🚀 Features

- **RTen Integration** — Direct adapter implementing `rusto_core::engine::InferenceEngine` and `InferenceSession`.
- **Format Support** — Supports both `.rten` flat model files and `.onnx` Open Neural Network Exchange formats.
- **Dynamic Engine Registry** — Registers seamlessly with `rusto-core` via `rusto_rten::register()`.
- **Zero C++ Dependencies** — Fully pure Rust execution without external C/C++ runtimes.

---

## 📦 Installation

Add `rusto-rten` to your `Cargo.toml`:

```toml
[dependencies]
rusto-rten = "0.4"
rusto-core = "0.4"
```

---

## 💡 Usage

```rust
use rusto_core::engine::EngineRegistry;
use rusto_rten::RtenEngine;
use std::sync::Arc;

// Register RTen engine with the global registry
rusto_rten::register();

// Or instantiate directly
let engine = Arc::new(RtenEngine::new());
let session = engine.create_session("model.rten")?;
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

