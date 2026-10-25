# @rustors/node

[![npm](https://img.shields.io/npm/v/@rustors/node.svg?logo=node.js&logoColor=white&color=339933)](https://www.npmjs.com/package/@rustors/node)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust Optical Character Recognition (OCR) engine for Node.js via native N-API addons.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

## Features

- **Blazing Fast N-API**: Zero IPC overhead, directly invoking RustO's pure-Rust engine.
- **RTen & ONNX Support**: High performance text detection and recognition.
- **Structured & Spatial Outputs**: Extracts bounding boxes, confidence scores, and physical 2D layouts.
- **Zero C++ Dependencies**: Compiles cleanly with standard Node.js toolchains.

## Installation

```bash
npm install @rustors/node
```

## Quick Start

```javascript
const { RustO } = require('@rustors/node');

// Initialize with model weights or JSON config
const engine = new RustO(JSON.stringify({
  detection: { model_path: 'path/to/det.rten' },
  recognition: { model_path: 'path/to/rec.rten', dict_path: 'path/to/dict.txt' }
}));

// Run OCR on an image file
const results = engine.detectTextFile('invoice.png', JSON.stringify({}));
results.forEach(res => {
  console.log(`${res.text} (${(res.score * 100).toFixed(1)}%)`);
});

// Run spatial OCR (reconstruct 2D layout)
const spatialText = engine.detectTextFileSpatial('invoice.png', JSON.stringify({}));
console.log(spatialText);
```

## 📖 Documentation

For full documentation, guides, and API reference:
- [Node.js Guide](https://byrizki.com/rusto-rs/how-to/use-node)
- [Full Documentation](https://byrizki.com/rusto-rs/)

## License

MIT License

