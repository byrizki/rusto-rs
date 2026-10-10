# @rustors/web

> Pure Web and Browser OCR library for [RustO!](https://github.com/byrizki/rusto-rs) powered by PaddleOCR models.

## Features

- **Canonical API**: 100% consistent with RustO across Android, iOS, .NET, and React Native.
- **Multiple Source Types**: Works seamlessly with URLs, Base64 strings, `Blob`, `File`, `HTMLImageElement`, `HTMLCanvasElement`, `ImageData`, `ImageBitmap`, or byte buffers.
- **Off-Thread Execution**: Web Worker support keeps the browser UI thread running smoothly at 60fps.
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

## License

MIT License
