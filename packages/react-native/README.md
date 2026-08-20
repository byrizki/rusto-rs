# React Native RustO

React Native bindings for the RustO! OCR library.

## Installation

```bash
npm install react-native-rusto
# or
yarn add react-native-rusto
# or
pnpm add react-native-rusto
```

### 1. iOS Setup

In your `ios/Podfile`, choose and add the model package you want to bundle (e.g. PP-OCRv6 Tiny, Small, Medium, PP-OCRv5, or PP-OCRv4):

```ruby
target 'YourApp' do
  # ... other pods

  # Add your preferred OCR models:
  pod 'RustO-Models-PPOCRv6-Tiny'    # ~6 MB (recommended default)
  # or pod 'RustO-Models-PPOCRv6-Small'   # ~30 MB
  # or pod 'RustO-Models-PPOCRv6-Medium'  # ~134 MB
  # or pod 'RustO-Models-PPOCRv5-Mobile'  # ~28 MB
  # or pod 'RustO-Models-PPOCRv4-Mobile'  # ~23 MB
end
```

Then install the CocoaPods:
```bash
cd ios && pod install
```

In your `android/build.gradle` (or `settings.gradle`), ensure `jitpack.io` repository is added:

```groovy
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

In your `android/app/build.gradle`, add the corresponding model package dependency under `dependencies`:

```groovy
dependencies {
    // Add your preferred OCR models:
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.4'     // ~6 MB (recommended default)
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-small:v0.2.4'    // ~30 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-medium:v0.2.4'   // ~134 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv5-mobile:v0.2.4'   // ~28 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv4-mobile:v0.2.4'   // ~23 MB
}
```

> **Note:** If you want to use custom models or load models dynamically from device storage, you don't need to install any model package. You can pass absolute filesystem paths directly to `initialize()`.

## Usage

```typescript
import { detectText, initialize } from 'react-native-rusto';

await initialize(); // PP-OCRv6 bundled defaults

const lines = await detectText({ uri: '/absolute/path/receipt.jpg' });
const words = await detectText(
  { uri: 'file:///absolute/path/receipt.jpg' },
  { output: 'words', lineYThreshold: 0.5, wordXThreshold: 0.4 },
);
const spatial = await detectText(
  { base64: encodedImage },
  { output: 'spatial', lineYThreshold: 0.5, wordXThreshold: 0.4 },
);
const fromBytes = await detectText({ bytes: new Uint8Array(imageBytes) });
```

`detectText` accepts exactly one source: `{ uri }`, `{ base64 }`, or `{ bytes }` (`Uint8Array` / `ArrayBuffer`). `{ uri }` accepts an absolute filesystem path, `file:` URI, or Android `content://` URI.

`output` defaults to `lines`, returning `TextResult[]`. `words` returns word boxes. `spatial` returns a formatted string. `lineYThreshold` groups vertically aligned regions; `wordXThreshold` controls word and spatial gaps.

Static model setup belongs in `initialize`:

```typescript
await initialize({
  preset: 'ppv6',
  models: { detection: 'det.mnn', recognition: 'rec.mnn', dictionary: 'dict.txt' },
});

const words = await detectText({ uri: '/absolute/path/receipt.jpg' }, {
  output: 'words',
  preprocessing: {
    maxSideLen: 1600,
    detection: { postprocess: { useDilation: true } },
  },
});
```

## Model Bundling

Model files can be bundled with your app. See [BUNDLING.md](./BUNDLING.md) for detailed instructions on how to:
- Bundle models with Android AAR
- Bundle models with iOS XCFramework
- Reduce app size with on-demand loading

## API

### `initialize(config?: InitializeConfig): Promise<void>`

Loads static OCR model resources. `preset` defaults to `ppv6`. Optional `models` keys are `detection`, `recognition`, `dictionary`, `classification`, and `orientation`. Image preprocessing is configured per request via `detectText`.

`preprocessing` belongs to `detectText` options, never `initialize`. It is request-local and does not mutate engine state. It contains resize/padding (`minHeight`, `maxSideLen`, `minSideLen`, `widthHeightRatio`), detector preprocessing (`limitSideLen`, `limitType`, `mean`, `std`), and detector postprocess (`threshold`, `boxThreshold`, `maxCandidates`, `unclipRatio`, `useDilation`).

### `detectText(source, options?)`

Runs one request. Source contains exactly one of `uri`, `base64`, or `bytes`. `uri` accepts an absolute filesystem path, `file:` URI, or Android `content://` URI. Options include `output: 'lines' | 'words' | 'spatial'`, `lineYThreshold`, `wordXThreshold`, `textScore`, `classification`, and `orientation`.

`lines` and `words` return `TextResult[]`; `spatial` returns `string`.

## Platform Support

- ✅ iOS 13.0+
- ✅ Android API 21+
- Architectures:
  - iOS: arm64, x86_64 (simulator)
  - Android: armeabi-v7a, arm64-v8a, x86, x86_64

## License

MIT
