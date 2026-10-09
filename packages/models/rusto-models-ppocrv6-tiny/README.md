# rusto-models-ppocrv6-tiny

> Pre-trained PP-OCRv6 Tiny model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `tiny`
- **Target Language:** `multilingual`
- **Version:** `0.3.1`

## Installation

```bash
npm install rusto-models-ppocrv6-tiny
# or
yarn add rusto-models-ppocrv6-tiny
# or
pnpm add rusto-models-ppocrv6-tiny
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import { modelMetadata, getModelUrls } from 'rusto-models-ppocrv6-tiny';
import { initialize } from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/rusto-models-ppocrv6-tiny');

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
import { modelMetadata, getModelRelativePaths } from 'rusto-models-ppocrv6-tiny';

console.log(modelMetadata.files);
```

## License

MIT License
