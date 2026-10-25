# @rustors/web

[![npm](https://img.shields.io/npm/v/@rustors/web.svg?logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/@rustors/web)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Pure Web and Browser OCR library for [RustO!](https://github.com/byrizki/rusto-rs) powered by PaddleOCR models.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

## Features

- **Canonical API**: 100% consistent with RustO across Android, iOS, .NET, and React Native.
- **Multiple Source Types**: Works with URLs, Base64 strings, `Blob`, `File`, `HTMLImageElement`, `HTMLCanvasElement`, `ImageData`, `ImageBitmap`, or byte buffers.
- **Off-Thread Execution**: Web Worker support keeps inference off the main browser UI thread.
- **Flexible Output**: Supports `'lines'`, `'words'`, and formatted `'spatial'` text outputs.
- **Zero Dependencies**: Lightweight with full TypeScript definitions.

## Installation

```bash
npm install @rustors/web
# or
yarn add @rustors/web
# or
pnpm add @rustors/web
```

## Quick Start

```ts
import { initialize, detectText } from '@rustors/web';
import { getModelUrls } from '@rustors/model-ppocrv6-tiny';

// 1. Initialize with pre-trained models
const urls = getModelUrls('/models/ppocrv6-tiny');

await initialize({
  preset: 'ppv6',
  models: {
    detection: urls.detection,
    recognition: urls.recognition,
    dictionary: urls.dictionary,
  },
});

// 2. Perform OCR on an image element, File, Blob, or URL
const results = await detectText(document.getElementById('my-image') as HTMLImageElement);

for (const result of results) {
  console.log(`${result.text} (score: ${result.score})`);
}

// 3. Or obtain formatted spatial text layout
const spatialText = await detectText('https://example.com/receipt.jpg', {
  output: 'spatial',
});
console.log(spatialText);
```

## 📖 Documentation

For full documentation, guides, and API reference:
- [Web (WASM) Guide](https://byrizki.com/rusto-rs/how-to/use-web)
- [Full Documentation](https://byrizki.com/rusto-rs/)

## License

MIT License

