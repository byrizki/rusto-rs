# rusto-models-ppocrv6-small

> Pre-trained PP-OCRv6 Small model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `small`
- **Target Language:** `multilingual`
- **Version:** `0.2.5`

## Installation

```bash
npm install rusto-models-ppocrv6-small
# or
yarn add rusto-models-ppocrv6-small
# or
pnpm add rusto-models-ppocrv6-small
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import { modelMetadata, getModelUrls } from 'rusto-models-ppocrv6-small';
import { initialize } from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/rusto-models-ppocrv6-small');

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
import { modelMetadata, getModelRelativePaths } from 'rusto-models-ppocrv6-small';

console.log(modelMetadata.files);
```

## License

MIT License
