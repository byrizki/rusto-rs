# @rustors/model-orient

> Pre-trained Document Orientation Classifier model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `pipeline`
- **Model Tier:** `optional`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-orient
# or
yarn add @rustors/model-orient
# or
pnpm add @rustors/model-orient
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-orient';

// Retrieve model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/orient');
console.log('Model URLs:', urls);
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rustors/model-orient';

console.log(modelMetadata.files);
```

## License

MIT License
