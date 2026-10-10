# @rusto/model-ppocrv6-medium-int8

> Pre-trained PP-OCRv6 Medium (INT8) model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `medium-int8`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rusto/model-ppocrv6-medium-int8
# or
yarn add @rusto/model-ppocrv6-medium-int8
# or
pnpm add @rusto/model-ppocrv6-medium-int8
```

## Usage

### In Web / React apps (`@rusto/web` or `@rusto/react`)

```ts
import { modelMetadata, getModelUrls } from '@rusto/model-ppocrv6-medium-int8';
import { initialize } from '@rusto/web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/ppocrv6-medium-int8');

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
import { modelMetadata, getModelRelativePaths } from '@rusto/model-ppocrv6-medium-int8';

console.log(modelMetadata.files);
```

## License

MIT License
