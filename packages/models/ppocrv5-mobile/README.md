# @rusto/model-ppocrv5-mobile

> Pre-trained PP-OCRv5 Mobile model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv5`
- **Model Tier:** `mobile`
- **Target Language:** `multilingual`
- **Version:** `0.3.1`

## Installation

```bash
npm install @rusto/model-ppocrv5-mobile
# or
yarn add @rusto/model-ppocrv5-mobile
# or
pnpm add @rusto/model-ppocrv5-mobile
```

## Usage

### In Web / React apps (`@rusto/web` or `@rusto/react`)

```ts
import { modelMetadata, getModelUrls } from '@rusto/model-ppocrv5-mobile';
import { initialize } from '@rusto/web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/ppocrv5-mobile');

await initialize({
  preset: 'ppv5',
  models: {
    detection: urls.detection,
    recognition: urls.recognition,
    dictionary: urls.dictionary,
  },
});
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rusto/model-ppocrv5-mobile';

console.log(modelMetadata.files);
```

## License

MIT License
