# @rustors/react-native

[![npm](https://img.shields.io/npm/v/@rustors/react-native.svg?logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/@rustors/react-native)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

Official React Native SDK (iOS & Android, Legacy & New Architecture) for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the pure-Rust MNN runtime (`rusto-nn`).

## Quick Overview

```ts
import { close, detectText, initialize } from '@rustors/react-native';

await initialize(); // Zero-config: auto-downloads Indra OCR models from Hugging Face on first run

const lines = await detectText({ uri })
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

const spatial = await detectText({ uri })
  .asSpatial()
  .run();

await close();
```

## Installation

```bash
npm install @rustors/react-native
# or
yarn add @rustors/react-native
```

### Android Repository Setup

`@rustors/react-native` resolves the native `com.github.byrizki.rusto-rs:rusto-android:v0.4.0` AAR from JitPack. Ensure JitPack is registered in your app's `android/build.gradle` (or `android/settings.gradle`):

```groovy
allprojects {
    repositories {
        mavenCentral()
        google()
        maven { url 'https://jitpack.io' }
    }
}
```

### iOS Setup

```bash
cd ios && pod install
```

> **Expo Note**: Native Rust binaries cannot run inside Expo Go. Use an Expo Development Build (`npx expo prebuild` + `npx expo run:ios` / `npx expo run:android`).

## Model Provisioning: Zero-Config vs. Offline Bundling

1. **Zero-Config Auto-Download (Default)**: Calling `await initialize()` with no arguments (or `await initialize({ model: 'ppocrv6-tiny', lang: 'latin' })`) automatically downloads the required `.mnn` models (or `.rten`) and dictionary from Hugging Face Hub (`byrizki/rten-models`) on first launch and caches them in the app's local cache directory (`cacheDir/rusto/models` on Android, `Caches/rusto/models` on iOS). No separate npm model packages, CocoaPods model specs, or Gradle model artifacts are required.
2. **Offline Bundled Assets**: For 100% air-gapped mobile apps, bundle `.mnn` (or `.rten`) and `dict.txt` (or `indra_meta.json`) files inside Android `assets/models/` and iOS Xcode Bundle Resources, then reference them via `initialize({ models: { ... } })`. See [**BUNDLING.md**](./BUNDLING.md) for step-by-step instructions.

## Complete Example

```tsx
import {
  close,
  detectText,
  initialize,
  type TextResult,
} from '@rustors/react-native';

export async function scanInvoice(uri: string): Promise<{
  lines: TextResult[];
  spatial: string;
}> {
  await initialize({
    preset: 'indra',
    enableReadingOrder: true,
  });

  try {
    // 1. Structured line-level recognition with UVDoc unwarping & layout reading order
    const lines = await detectText({ uri })
      .withAutoUnwarp()
      .withLayoutReading()
      .withPreDetection({ targetMaxSide: 1600 })
      .withDetection({ limitSideLen: 960, limitType: 'max' })
      .withPostDetection({ useDilation: true, unclipRatio: 1.8 })
      .asLines()
      .run();

    // 2. Reconstructed 2D spatial plain-text layout
    const spatial = await detectText({ uri })
      .withPreRecognition({ lineYThreshold: 0.5, wordXThreshold: 0.4 })
      .asSpatial()
      .run();

    return { lines, spatial };
  } finally {
    await close();
  }
}
```

## Public API

### `initialize(config?: InitializeConfig): Promise<void>`

Initializes or replaces the native OCR session.

```ts
// Zero-config Indra OCR default (auto-downloads models from Hugging Face on first run)
await initialize();

// Offline bundled .mnn / .rten assets (Android assets/models/* & iOS bundle resources)
await initialize({
  preset: 'ppv6',
  models: {
    detection: 'models/ppocrv6_tiny_det.mnn',
    recognition: 'models/ppocrv6_tiny_rec.mnn',
    dictionary: 'models/ppocrv6_dict.txt',
    unwarp: 'models/uvdoc.mnn',
    orientation: 'models/ppocr_orient.mnn',
    layout: 'models/layout_cdla.mnn',
  },
});
```

| Field | Values | Description |
| :--- | :--- | :--- |
| `preset` | `'indra'`, `'ppv6'`, `'ppv5'`, `'ppv4'`, `'ppv3'` | Architecture preset |
| `model` | `'indraocr'`, `'ppocrv6-tiny'`, `'ppocrv6-small'`, `'ppocrv5-mobile'`, etc. | Canonical pipeline name for auto-download |
| `lang` | `'latin'`, `'cjk'`, `'ja'`, `'korean'`, `'ar'`, `'cyrillic'`, etc. | Target recognition script/language |
| `modelPath` / `cacheDir` | `string` | Custom model directory or writable download cache directory |
| `enableReadingOrder` | `boolean` | Load neural reading-order transformer |
| `layoutVariant` | `'cdla'` \| `'publaynet'` | Document layout model taxonomy |
| `recBatchNum` | integer `>= 1` | Recognition batch size override |
| `models` | `{ detection, recognition, dictionary, classification, orientation, unwarp, layout }` | Explicit bundled or local `.mnn` / `.rten` / `.txt` asset paths |

### `detectText(source)` -> `DetectTextBuilder`

`source` must contain **exactly one** of:

| Source | Type | Notes |
| :--- | :--- | :--- |
| `{ uri }` | `string` | Local path, `file://` URI, or Android `content://` URI |
| `{ base64 }` | `string` | Base64-encoded PNG/JPEG/WebP payload |
| `{ bytes }` | `Uint8Array \| ArrayBuffer` | Encoded image bytes |

Chain pipeline stages and output granularity before calling `.run()`:
- **Output Granularity**: `.asLines()`, `.asWords()`, `.asSpatial()`, or `.withOutput('lines' | 'words' | 'spatial')`
- **Pipeline Toggles**: `.withAutoUnwarp()`, `.withLayoutReading()`, `.withOrientationDetection()`, `.withLineClassification()`
- **6-Phase Tuning**: `.withPreDetection(...)`, `.withDetection(...)`, `.withPostDetection(...)`, `.withPreRecognition(...)`, `.withRecognition(...)`, `.withPostRecognition(...)`
- **Terminal Execution**: `.run()`

### `close(): Promise<void>`

Releases the active native OCR engine instance and frees model memory.

## Result Types

### `TextResult` (from `.asLines().run()` and `.asWords().run()`)

```ts
interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  frame: { width: number; height: number; top: number; left: number };
  region_id?: number;
  layout_type?: string;
}
```

### Spatial String (from `.asSpatial().run()`)

Returns a `string` preserving approximate columns and line spacing.

## Platform Support

- **iOS**: 13.0+ (arm64 device & arm64/x86_64 simulator)
- **Android**: API 21+ (`arm64-v8a`, `armeabi-v7a`, `x86_64`)
- **React Native**: Both Legacy Bridge and New Architecture (TurboModules)

## License

MIT
