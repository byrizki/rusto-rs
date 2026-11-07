# rusto-node 🟢

[![npm (node)](https://img.shields.io/npm/v/@rustors/node.svg?logo=node.js&logoColor=white&color=339933)](https://www.npmjs.com/package/@rustors/node)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Node.js native addon for [RustO!](https://github.com/byrizki/rusto-rs) powered by N-API (`napi-rs`).

`rusto-node` provides zero-overhead, multi-threaded text extraction, document unwarping, layout reading order analysis, and spatial reconstruction directly to Node.js and TypeScript applications. Powered by `rusto-nn`, it runs without external C++ or OpenCV dependencies.

---

## 🚀 Features

- **Native N-API Performance** — Compiled directly to native Node.js binary addons with zero runtime overhead and asynchronous event loop safety.
- **Pure-Rust Inference (`rusto-nn`)** — Pure Rust neural network execution with AVX-512, AVX2, and NEON SIMD acceleration.
- **Multi-Input Support** — Process images directly from file paths, Node.js `Buffer` objects, or raw RGB / RGBA byte buffers.
- **Auto-Model Download** — Automatically caches and verifies official models (Indra OCR, PP-OCRv6, PP-OCRv5, PP-OCRv4) from Hugging Face Hub.
- **Complete Pipeline Control** — Supports 3D document unwarping, orientation correction, layout classification, and spatial reading order.

---

## 📦 Installation

```bash
npm install @rustors/node
# or
pnpm add @rustors/node
# or
yarn add @rustors/node
```

### Building from Source

```bash
cd crates/rusto-node
npm run build
```

---

## 💡 Quick Start

```typescript
import { RustO } from '@rustors/node';

// Initialize RustO with default models (auto-downloads if needed)
const config = JSON.stringify({
  model: 'indra',
  lang: 'en'
});

const ocr = RustO.initialize(config);

// Run text detection on an image file
const resultJson = ocr.detectTextFile('receipt.jpg', JSON.stringify({
  autoUnwarp: true,
  layoutReading: true,
  output: 'Lines'
}));

const results = JSON.parse(resultJson);
console.log('Detected text:', results);
```

### Processing from a Buffer

```typescript
import * as fs from 'fs';
import { RustO } from '@rustors/node';

const ocr = RustO.initialize('{}');
const buffer = fs.readFileSync('document.png');

const resultJson = ocr.detectTextData(buffer, JSON.stringify({
  output: 'Words'
}));

console.log(JSON.parse(resultJson));
```

---

## 📄 License

Licensed under the [MIT License](../../LICENSE-MIT).

