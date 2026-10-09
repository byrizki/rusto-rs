# rusto-models-ppocrv6-medium

> Pre-trained PP-OCRv6 Medium model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `ppv6`
- **Model Tier:** `medium`
- **Target Language:** `multilingual`
- **Version:** `0.3.1`

## Installation

```bash
npm install rusto-models-ppocrv6-medium
# or
yarn add rusto-models-ppocrv6-medium
# or
pnpm add rusto-models-ppocrv6-medium
```

## Usage

### In Web / React apps (`rusto-web` or `react-rusto`)

```ts
import { modelMetadata, getModelUrls } from 'rusto-models-ppocrv6-medium';
import { initialize } from 'rusto-web';

// Initialize with model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/rusto-models-ppocrv6-medium');

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
import { modelMetadata, getModelRelativePaths } from 'rusto-models-ppocrv6-medium';

console.log(modelMetadata.files);
```

## License

MIT License
