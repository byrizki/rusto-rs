# @rusto/model-ppocrv6-tiny

> Pre-trained PP-OCRv6 Tiny model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `tiny`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rusto/model-ppocrv6-tiny
# or
yarn add @rusto/model-ppocrv6-tiny
# or
pnpm add @rusto/model-ppocrv6-tiny
```

## Usage

### In Web / React apps (`@rusto/web` or `@rusto/react`)

```ts
import { modelMetadata, getModelUrls } from '@rusto/model-ppocrv6-tiny';
import { initialize } from '@rusto/web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/ppocrv6-tiny');

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
import { modelMetadata, getModelRelativePaths } from '@rusto/model-ppocrv6-tiny';

console.log(modelMetadata.files);
```

## License

MIT License
