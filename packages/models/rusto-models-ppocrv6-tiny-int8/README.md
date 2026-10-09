# rusto-models-ppocrv6-tiny-int8

> Pre-trained PP-OCRv6 Tiny (INT8) model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `tiny-int8`
- **Target Language:** `multilingual`
- **Version:** `0.3.1`

## Installation

```bash
npm install rusto-models-ppocrv6-tiny-int8
# or
yarn add rusto-models-ppocrv6-tiny-int8
# or
pnpm add rusto-models-ppocrv6-tiny-int8
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import { modelMetadata, getModelUrls } from 'rusto-models-ppocrv6-tiny-int8';
import { initialize } from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/rusto-models-ppocrv6-tiny-int8');

await initialize({
  preset: 'ppv6',
  models: {
    detection: urls.detection,
    recognition: urls.recognition,
    dictionary: urls.dictionary,
  },
});
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from 'rusto-models-ppocrv6-tiny-int8';

console.log(modelMetadata.files);
```

## License

MIT License
