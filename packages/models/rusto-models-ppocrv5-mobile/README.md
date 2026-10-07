# rusto-models-ppocrv5-mobile

> Pre-trained PP-OCRv5 Mobile MNN model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv5`
- **Model Tier:** `mobile`
- **Target Language:** `multilingual`
- **Version:** `0.2.5`

## Installation

```bash
npm install rusto-models-ppocrv5-mobile
# or
yarn add rusto-models-ppocrv5-mobile
# or
pnpm add rusto-models-ppocrv5-mobile
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import { modelMetadata, getModelUrls } from 'rusto-models-ppocrv5-mobile';
import { initialize } from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/rusto-models-ppocrv5-mobile');

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
import { modelMetadata, getModelRelativePaths } from 'rusto-models-ppocrv5-mobile';

console.log(modelMetadata.files);
```

## License

MIT License
