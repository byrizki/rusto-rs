# rusto-wasm 🌐

[![npm (web)](https://img.shields.io/npm/v/@rustors/web.svg?logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/@rustors/web)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

WebAssembly (WASM) engine for [RustO!](https://github.com/byrizki/rusto-rs) — delivering client-side, privacy-preserving document OCR and text detection directly in modern browsers and edge runtimes.

`rusto-wasm` brings pure-Rust deep learning inference (`rusto-nn`) into the browser with zero native runtime dependencies, optional WebGPU compute acceleration, and support for MNN, ONNX, and RTen model formats.

---

## 🚀 Features

- **100% Client-Side OCR** — Process images, documents, and camera streams locally in the browser without transmitting sensitive user data to servers.
- **Pure-Rust MNN & Multi-Format Engine** — Built on `rusto-nn` with default support for MNN (`.mnn`), ONNX (`.onnx`), and RTen (`.rten`) model formats.
- **WebGPU Compute Acceleration** — Optional GPU-accelerated inference via `rusto-nn-webgpu` utilizing modern browser WebGPU compute pipelines.
- **Optimized for WASM Performance** — Unrolled WASM SIMD128 kernels, aspect-ratio crop clustering, and streaming batch inference.
- **Fluent JS / TS API** — Structured text results with bounding boxes, confidence scores, spatial text reconstruction, and TSV formatting.

---

## 📦 Building and Packaging

Build the WASM package using [`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```bash
# Build for modern bundlers (Webpack, Vite, Rollup)
wasm-pack build crates/rusto-wasm --target web --release

# Build with WebGPU acceleration
wasm-pack build crates/rusto-wasm --target web --release --features rusto-webgpu
```

### Feature Flags

| Feature | Default | Description |
|---|---|---|
| `rusto-nn` | **Yes** | Pure Rust inference engine (`rusto-nn`) |
| `onnx` | **Yes** | Support for loading `.onnx` models |
| `rten-format` | **Yes** | Support for loading `.rten` models |
| `rusto-webgpu` | No | Enables WebGPU compute shader acceleration |
| `rusto-rten` | No | RTen engine backend alternative |

---

## 💡 JavaScript / TypeScript Usage

```javascript
import init, { RtenModel, wasm_is_ready, wasm_engine_backend } from './pkg/rusto_wasm.js';

async function run() {
  await init();
  console.log('Engine ready:', wasm_is_ready());
  console.log('Active backend:', wasm_engine_backend());

  // Fetch model weights
  const modelBytes = await fetch('/models/indra_rec.mnn').then(res => res.arrayBuffer());
  const model = RtenModel.from_bytes(new Uint8Array(modelBytes));

  // Run inference
  const inputTensor = new Float32Array(1 * 3 * 48 * 320);
  const output = model.run(inputTensor, [1, 3, 48, 320]);
  console.log('Output shape:', output.shape);
}

run();
```

---

## 📄 License

Dual-licensed under either of:
- [Apache License, Version 2.0](../../LICENSE-APACHE)
- [MIT License](../../LICENSE-MIT)

