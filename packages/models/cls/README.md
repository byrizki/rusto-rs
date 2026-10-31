# @rustors/model-cls

> Pre-trained Text-Line Angle Classifier model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `pipeline`
- **Model Tier:** `optional`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-cls
# or
yarn add @rustors/model-cls
# or
pnpm add @rustors/model-cls
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-cls';

// Retrieve model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/cls');
console.log('Model URLs:', urls);
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rustors/model-cls';

console.log(modelMetadata.files);
```

## License

MIT License
