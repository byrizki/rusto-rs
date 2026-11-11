# @rustors/node

Official Node.js N-API native bindings for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and pure-Rust MNN (`rusto-nn`) inference.

## Features

- **Zero-Config PP-OCRv6 Tiny Default**: Initialize with `RustO.initialize()` — automatically downloads and caches the default `ppocrv6-tiny` (FP32, Apache 2.0) pipeline on first run from Hugging Face Hub.
- **Pure-Rust Native Addon**: Zero C++ or Python runtime dependencies; prebuilt N-API binaries for Linux, macOS, and Windows (`x64` & `arm64`).
- **Document Intelligence Pipeline**: Built-in UVDoc document unwarping (`withAutoUnwarp()`), 4-way page orientation correction (`withOrientationDetection()`), layout analysis, and neural reading-order recovery (`withLayoutReading()`).
- **Fluent Builder API**: Configure the 6-phase OCR pipeline and select `.asLines()`, `.asWords()`, `.asSpatial()`, or `.asTsv()` before calling `.run()`.
- **Flexible Image Inputs**: Pass file paths, `Buffer` / `Uint8Array`, `{ path }`, `{ uri }`, `{ bytes }`, or raw interleaved RGB8 buffers via `.rgb(...)`.

## Installation

```bash
npm install @rustors/node
# or
pnpm add @rustors/node
```

## Quick Start (Zero-Config)

Calling `RustO.initialize()` with no arguments initializes the default **PP-OCRv6 Tiny (FP32)** pipeline and automatically downloads missing model assets from Hugging Face Hub into your user cache directory (`~/.cache/rusto/models` or `$RUSTO_CACHE_DIR`).

```ts
import { RustO } from '@rustors/node';

const engine = RustO.initialize();

try {
  // Structured line-level recognition with unwarping & layout reading order
  const lines = engine
    .detectText('invoice.png')
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run();

  for (const item of lines) {
    console.log(
      `[${item.score.toFixed(2)}] (${item.layout_type ?? 'line'} #${item.region_id ?? '-'}) ${item.text}`
    );
  }

  // Reconstructed 2D spatial text layout
  const spatialLayout = engine
    .detectText('invoice.png')
    .asSpatial()
    .run();

  console.log(spatialLayout);
} finally {
  engine.close();
}
```

## Custom Initialization & Presets

You can pass an `InitializeConfig` object (or JSON string) to `RustO.initialize(config)` or `new RustO(config)` to select a model family, target language, or local `.mnn` / `.rten` files:

```ts
import { RustO, type InitializeConfig } from '@rustors/node';

// Option A: Select a canonical model & language (auto-downloaded if not cached)
const fastEngine = RustO.initialize({
  model: 'ppocrv6-tiny',
  lang: 'latin',
  enableReadingOrder: true,
});

// Option B: Explicit offline model paths via constructor or initialize()
const offlineConfig: InitializeConfig = {
  preset: 'indra',
  enableReadingOrder: true,
  detection: {
    modelPath: './models/indra_det.mnn',
  },
  recognition: {
    modelPath: './models/indra_rec.mnn',
    dictPath: './models/indra_meta.json',
  },
  unwarp: {
    enabled: true,
    modelPath: './models/uvdoc.mnn',
  },
  layout: {
    enabled: true,
    modelPath: './models/layout_cdla.mnn',
  },
};

const offlineEngine = new RustO(offlineConfig);
```

### Pre-Downloading Models Ahead of Time

For containerized deployments or air-gapped builds, pre-warm the model cache during build time:

```ts
import { RustO } from '@rustors/node';

// Download default PP-OCRv6 Tiny assets (or pass 'ppocrv6-tiny-fp16', 'indraocr', 'ppocrv5-mobile', etc.)
const cachedPaths = RustO.downloadModel('ppocrv6-tiny', undefined, './models-cache');
console.log('Downloaded assets:', cachedPaths);
```

## Input Sources & Raw RGB Buffers

`engine.detectText(source)` accepts file paths, encoded image `Buffer` / `Uint8Array` payloads, or raw 24-bit interleaved RGB pixel buffers:

```ts
import { readFileSync } from 'node:fs';
import { RustO } from '@rustors/node';

const engine = RustO.initialize();

// 1. Encoded PNG/JPEG/WebP buffer
const pngBytes = readFileSync('receipt.png');
const wordItems = engine.detectText(pngBytes).asWords().run();

// 2. Raw interleaved RGB8 buffer (e.g. from sharp or video frames)
const width = 1280;
const height = 720;
const stride = width * 3; // optional; defaults to width * 3
const rgbBuf = Buffer.alloc(stride * height);

const rgbResults = engine
  .detectText()
  .rgb(rgbBuf, width, height, stride)
  .asLines()
  .run();

engine.close();
```

## 6-Phase Pipeline Tuning

Fine-tune individual stages per request on `DetectTextBuilder`:

```ts
const results = engine
  .detectText('scan.jpg')
  .withOrientationDetection()
  .withAutoUnwarp()
  .withLayoutReading()
  .withPreDetection({ descreenStrength: 0.4, targetMaxSide: 1280 })
  .withDetection({ limitSideLen: 960, limitType: 'max' })
  .withPostDetection({ threshold: 0.3, boxThreshold: 0.6, unclipRatio: 1.6 })
  .withPreRecognition({ lineYThreshold: 0.5, wordXThreshold: 0.4 })
  .withRecognition({ batchSize: 8 })
  .withPostRecognition({ scoreThreshold: 0.5 })
  .withOutput('lines') // 'lines' | 'words' | 'spatial'
  .run();
```

## Output Types

### `TextResult` (returned by `.asLines().run()` and `.asWords().run()`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `text` | `string` | Recognized UTF-8 text string |
| `score` | `number` | Recognition confidence score (`0.0` – `1.0`) |
| `box_points` | `[[number, number], ...]` | 4-point bounding polygon `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` |
| `frame` | `{ width, height, top, left }` | Axis-aligned bounding box in image pixels |
| `region_id` | `number \| null \| undefined` | Document layout region index (`regionId` metadata when `withLayoutReading()` is enabled) |
| `layout_type` | `LayoutType \| string \| null \| undefined` | Semantic layout label (`layoutType`: `'text'`, `'title'`, `'table'`, `'header'`, `'footer'`, `'figure'`, etc.) |

### Spatial String (returned by `.asSpatial().run()`)

Returns a `string` preserving the 2D visual column and line spacing of the source document.

## License

Apache-2.0 OR MIT
