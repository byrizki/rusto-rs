# @rustors/model-uvdoc

> Pre-trained Document Unwarping (UVDoc) model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `pipeline`
- **Model Tier:** `optional`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-uvdoc
# or
yarn add @rustors/model-uvdoc
# or
pnpm add @rustors/model-uvdoc
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-uvdoc';

// Retrieve model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/uvdoc');
console.log('Model URLs:', urls);
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rustors/model-uvdoc';

console.log(modelMetadata.files);
```

## License

MIT License
