# rusto-nn-webgpu 🎮

[![Crates.io](https://img.shields.io/crates/v/rusto-nn-webgpu.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-nn-webgpu)
[![docs.rs](https://img.shields.io/docsrs/rusto-nn-webgpu?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-nn-webgpu)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Pluggable [WebGPU](https://www.w3.org/TR/webgpu/) hardware acceleration backend for [`rusto-nn`](../rusto-nn).

`rusto-nn-webgpu` accelerates deep learning inference across modern GPUs via `wgpu` and WGSL compute shaders. It functions identically across desktop environments (Vulkan, Metal, DirectX 12) and browser WebAssembly runtimes (`navigator.gpu`).

---

## 🚀 Features

- **Cross-Platform GPU Compute** — Supports Windows (DX12/Vulkan), macOS/iOS (Metal), Linux (Vulkan), and Browsers (WebGPU).
- **WGSL Fused Compute Shaders** — Custom tiled GEMM matrix multiplication shaders with fused ReLU, Sigmoid, and HardSwish activations.
- **Buffer Pooling & Re-use** — Amortized GPU memory allocation with reusable buffer pools for zero-allocation per-inference passes.
- **Browser WASM Native** — Implements asynchronous WebGPU device initialization for seamless use inside WebAssembly workers.
- **Pluggable Architecture** — Implements `rusto_nn::DeviceBackend` with dynamic runtime registration.

---

## 📦 Installation

Add `rusto-nn-webgpu` to your `Cargo.toml`:

```toml
[dependencies]
rusto-nn = "0.4"
rusto-nn-webgpu = "0.4"
```

---

## 💡 Usage

### Desktop (Synchronous)

```rust
use rusto_nn_webgpu::{WebGpuBackend, init_and_register};

if WebGpuBackend::is_webgpu_supported() {
    match init_and_register() {
        Ok(()) => println!("WebGPU acceleration initialized and registered!"),
        Err(e) => eprintln!("Failed to initialize WebGPU: {e}"),
    }
}
```

### Browser / WASM (Async)

```rust
#[cfg(target_arch = "wasm32")]
pub async fn enable_webgpu() -> Result<bool, String> {
    if !rusto_nn_webgpu::WebGpuBackend::is_webgpu_supported() {
        return Ok(false);
    }
    let backend = rusto_nn_webgpu::WebGpuBackend::init_async().await
        .map_err(|e| format!("{e:?}"))?;
    backend.register();
    Ok(true)
}
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

