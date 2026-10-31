# @rustors/model-layout-s

> Pre-trained Document Layout Detection (Small) model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `pipeline`
- **Model Tier:** `optional`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-layout-s
# or
yarn add @rustors/model-layout-s
# or
pnpm add @rustors/model-layout-s
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-layout-s';

// Retrieve model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/layout-s');
console.log('Model URLs:', urls);
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rustors/model-layout-s';

console.log(modelMetadata.files);
```

## License

MIT License
