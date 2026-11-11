# @rustors/web

Official Browser & WebAssembly SDK for **RustO!** — client-side, privacy-preserving OCR powered by PaddleOCR, Indra OCR, and the pure-Rust `rusto-nn` & RTen WebAssembly runtimes.

## Features

- **100% Client-Side WebAssembly**: Runs in a dedicated Web Worker by default so your main UI thread stays responsive and images never leave the browser.
- **Standalone Model Assets**: Load lightweight models directly from Hugging Face Hub (`byrizki/rusto-models`), your CDN, or local `/models/` static assets — no npm model wrapper packages required.
- **PP-OCRv6 Tiny Default & Presets**: Defaults to Apache 2.0 `'ppv6'` (`ppocrv6-tiny` FP32) alongside `'ppocrv6-tiny-fp16'`, `'ppv6'`, `'ppv5'`, `'ppv4'`, `'ppv3'`, and `'indra'`.
- **Document Intelligence**: Configure optional `unwarp`, `orientation`, `layout`, and `order` `.rten` models and enable them per call with `.withAutoUnwarp()` and `.withLayoutReading()`.
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, or `.withOutput(...)` into a single `await ...run()` call.

## Installation

```bash
npm install @rustors/web
# or
pnpm add @rustors/web
```

> Using React? Check out [`@rustors/react`](../react/README.md) for `<RustoProvider>`, `useOcr()`, `<OcrDropzone>`, and `<OcrOverlay>`.

## Hosting or Fetching Models

`@rustors/web` loads standalone model files via URL (from your `public/models/` directory, your CDN, or Hugging Face Hub `https://huggingface.co/byrizki/rusto-models/resolve/main/` with CORS enabled) or directly as `Uint8Array` / `ArrayBuffer`.

| Pipeline / Stage | Canonical Asset Filenames |
| :--- | :--- |
| **PP-OCRv6 Tiny (`preset: 'ppv6'`, Default)** | `ppocrv6-tiny/det.rten`, `ppocrv6-tiny/rec.rten`, `ppocrv6-tiny/dict.txt` |
| **Indra OCR (`preset: 'indra'`)** | `indraocr/det.rten`, `indraocr/rec.rten`, `indraocr/meta.json` |
| **Optional Document Stages** | `modules/uvdoc.rten` (`unwarp`), `modules/orient.rten` (`orientation`), `modules/layout_cdla.rten` (`layout`), `indraocr/order.rten` (`order`) |

## Quick Start

```ts
import { initialize, detectText, close } from '@rustors/web';

const HF_BASE = 'https://huggingface.co/byrizki/rusto-models/resolve/main';

// 1. Initialize once (runs inside a dedicated Web Worker by default)
await initialize({
  preset: 'ppv6',
  enableReadingOrder: true,
  models: {
    detection: `${HF_BASE}/ppocrv6-tiny/det.rten`,
    recognition: `${HF_BASE}/ppocrv6-tiny/rec.rten`,
    dictionary: `${HF_BASE}/ppocrv6-tiny/dict.txt`,
    // Optional auxiliary .rten models:
    unwarp: `${HF_BASE}/modules/uvdoc.rten`,
    orientation: `${HF_BASE}/modules/orient.rten`,
    layout: `${HF_BASE}/modules/layout_cdla.rten`,
    order: `${HF_BASE}/indraocr/order.rten`,
  },
});

// 2. Structured line-level recognition
const fileInput = document.querySelector<HTMLInputElement>('#invoice-input')!;
const file = fileInput.files![0];

const lines = await detectText(file)
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

for (const item of lines) {
  console.log(
    `[${item.score.toFixed(2)}] (${item.layout_type ?? 'line'} #${item.region_id ?? '-'}) ${item.text}`
  );
}

// 3. Reconstructed 2D spatial text layout
const spatialText = await detectText(file)
  .asSpatial()
  .run();

console.log(spatialText);

// 4. Release Web Worker when finished
close();
```

### Lightweight PP-OCRv6 Tiny Configuration (Self-Hosted Static Assets)

Place `.rten` files under `public/models/` in Vite / Next.js / Nuxt for fast, same-origin caching:

```ts
import { WebOcrEngine } from '@rustors/web';

const engine = new WebOcrEngine();
await engine.initialize({
  preset: 'ppv6',
  models: {
    detection: '/models/ppocrv6_tiny_det.rten',
    recognition: '/models/ppocrv6_tiny_rec.rten',
    dictionary: '/models/ppocrv6_dict.txt',
  },
});

const words = await engine.detectText(imageBlob).asWords().run();
engine.close();
```

## Supported `ImageSource` Inputs

`detectText(source)` accepts any standard browser image representation:
- `File` / `Blob`
- Image URL `string` or Data URL
- `HTMLImageElement`, `HTMLCanvasElement`, `HTMLVideoElement`, `ImageBitmap`, `OffscreenCanvas`
- `ImageData`
- `Uint8Array` / `ArrayBuffer`

## 6-Phase Pipeline Options

```ts
const results = await detectText(canvasElement)
  .withOrientationDetection()
  .withAutoUnwarp()
  .withLayoutReading()
  .withPreDetection({ descreenStrength: 0.3, targetMaxSide: 1280 })
  .withDetection({ limitSideLen: 960, limitType: 'max' })
  .withPostDetection({ threshold: 0.3, boxThreshold: 0.6, unclipRatio: 1.6 })
  .withPreRecognition({ lineYThreshold: 0.5, wordXThreshold: 0.4 })
  .withRecognition({ batchSize: 8 })
  .withPostRecognition({ scoreThreshold: 0.5 })
  .withOutput('lines')
  .run();
```

## Output Types

- `.asLines().run()` / `.asWords().run()` -> `Promise<TextResult[]>`:
  - `text`: Recognized string
  - `score`: Confidence (`0.0` – `1.0`)
  - `box_points`: 4-corner polygon `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`
  - `frame`: `{ width, height, top, left }`
  - `region_id`: Optional layout region index when `withLayoutReading()` is enabled
  - `layout_type`: Optional semantic layout category (`'text'`, `'title'`, `'table'`, `'header'`, `'footer'`, `'figure'`, etc.)
- `.asSpatial().run()` -> `Promise<string>`: Reconstructed 2D plain-text layout.

## License

Apache-2.0 OR MIT
