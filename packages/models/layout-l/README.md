# @rustors/model-layout-l

> Pre-trained Document Layout Detection (Large) model package for [RustO!](https://github.com/byrizki/rusto-rs)

## Metadata

- **Model Preset:** `pipeline`
- **Model Tier:** `optional`
- **Target Language:** `multilingual`
- **Version:** `0.3.2`

## Installation

```bash
npm install @rustors/model-layout-l
# or
yarn add @rustors/model-layout-l
# or
pnpm add @rustors/model-layout-l
```

## Usage

### In Web / React apps (`@rustors/web` or `@rustors/react`)

```ts
import { modelMetadata, getModelUrls } from '@rustors/model-layout-l';

// Retrieve model URLs (e.g. hosted under public/cdn)
const urls = getModelUrls('/models/layout-l');
console.log('Model URLs:', urls);
```

### In Node.js / Server environments

```ts
import { modelMetadata, getModelRelativePaths } from '@rustors/model-layout-l';

console.log(modelMetadata.files);
```

## License

MIT License
