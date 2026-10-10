# @rustors/model-ppocrv5-mobile

> Pre-trained PP-OCRv5 Mobile model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv5`
- **Model Tier:** `mobile`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-ppocrv5-mobile
# or
yarn add @rustors/model-ppocrv5-mobile
# or
pnpm add @rustors/model-ppocrv5-mobile
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-ppocrv5-mobile';
import { initialize } from '@rustors/web';

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
import { modelMetadata, getModelRelativePaths } from '@rustors/model-ppocrv5-mobile';

console.log(modelMetadata.files);
```

## License

MIT License
