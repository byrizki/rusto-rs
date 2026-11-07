# rusto-nn-cuda 🚀

[![Crates.io](https://img.shields.io/crates/v/rusto-nn-cuda.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-nn-cuda)
[![docs.rs](https://img.shields.io/docsrs/rusto-nn-cuda?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-nn-cuda)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Pluggable NVIDIA CUDA hardware acceleration backend for [`rusto-nn`](../rusto-nn).

`rusto-nn-cuda` enables GPU-accelerated tensor computations, fused matrix multiplications (GEMM), and convolution operations on NVIDIA GPUs using `cudarc` and cuBLAS.

---

## 🚀 Features

- **cuBLAS Acceleration** — High-throughput batched matrix multiplications and linear algebra kernels.
- **Dynamic Driver Loading** — Gracefully detects presence of CUDA driver runtime (`libcuda.so` / `nvcuda.dll`) with fallback to pure-Rust CPU execution if unavailable.
- **Seamless Engine Integration** — Plugs into `rusto-nn` via the `DeviceBackend` trait.
- **Zero Static CUDA Linking** — Uses dynamic loading via `cudarc` fallback dynamic loading, avoiding hard compile-time CUDA SDK dependencies.

---

## 📦 Requirements

- NVIDIA GPU with Compute Capability 5.0+ (Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell)
- NVIDIA CUDA Driver installed (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows)

---

## 📦 Installation

Add `rusto-nn-cuda` to your `Cargo.toml`:

```toml
[dependencies]
rusto-nn = "0.4"
rusto-nn-cuda = "0.4"
```

---

## 💡 Usage

```rust
use rusto_nn_cuda::{CudaBackend, is_cuda_driver_available};

if is_cuda_driver_available() {
    match CudaBackend::init() {
        Ok(backend) => {
            // Register CUDA accelerator with rusto-nn
            backend.register();
            println!("CUDA acceleration enabled!");
        }
        Err(e) => eprintln!("Failed to initialize CUDA: {e}"),
    }
} else {
    println!("CUDA driver not detected, using pure-Rust CPU SIMD kernels.");
}
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

