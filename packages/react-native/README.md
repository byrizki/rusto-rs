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

### 2. Android Setup

In your `android/app/build.gradle`, add the corresponding model package dependency under `dependencies`:

```groovy
dependencies {
    // Add your preferred OCR models:
    implementation 'com.byrizki.rusto:rusto-models-ppocrv6-tiny:0.1.7'     // ~6 MB (recommended default)
    // or implementation 'com.byrizki.rusto:rusto-models-ppocrv6-small:0.1.7'    // ~30 MB
    // or implementation 'com.byrizki.rusto:rusto-models-ppocrv6-medium:0.1.7'   // ~134 MB
    // or implementation 'com.byrizki.rusto:rusto-models-ppocrv5-mobile:0.1.7'   // ~28 MB
    // or implementation 'com.byrizki.rusto:rusto-models-ppocrv4-mobile:0.1.7'   // ~23 MB
}
```

> **Note:** If you want to use custom models or load models dynamically from device storage, you don't need to install any model package. You can pass absolute filesystem paths directly to `initialize()`.

## Usage

### Basic Usage (with bundled models)

```typescript
import { initialize, detectText, getVersion } from 'react-native-rusto';

// Initialize with bundled default models (no parameters needed!)
await initialize();

// Detect text from image file
const results = await detectText('/path/to/image.jpg');

results.forEach((result) => {
  console.log(`Text: ${result.text}`);
  console.log(`Score: ${result.score}`);
  console.log(`Frame: ${result.frame.left}, ${result.frame.top}, ${result.frame.width}x${result.frame.height}`);
  console.log(`Box: ${JSON.stringify(result.box_points)}`);
});

// Get library version
const version = await getVersion();
console.log(`RustO version: ${version}`);
```

### Advanced Usage (with RustOConfig)

```typescript
import { initialize, detectText, detectTextToSpatialText } from 'react-native-rusto';

// Initialize with unified RustOConfig
await initialize({
  detModelPath: 'det.mnn',
  recModelPath: 'rec.mnn',
  dictPath: 'dict.txt',
  textScore: 0.6,
  detThresh: 0.3,
  detBoxThresh: 0.5,
  yThresholdMultiplier: 0.5,
  xThresholdMultiplier: 0.4,
});

// Or extract spatial text with custom XY thresholds
const spatialText = await detectTextToSpatialText('/path/to/image.jpg', 0.5, 0.4);
console.log(spatialText);
```

### Model Bundling

Model files can be bundled with your app. See [BUNDLING.md](./BUNDLING.md) for detailed instructions on how to:
- Bundle models with Android AAR
- Bundle models with iOS XCFramework
- Reduce app size with on-demand loading

## API

### `initialize(configOrDetModel?: RustOConfig | string, recModel?: string, dict?: string): Promise<boolean>`

Initialize the RustO engine with a `RustOConfig` object or individual model files.

**Parameters:**
- `configOrDetModel`: A `RustOConfig` configuration object, or path to detection model file (default: `'det.mnn'`)
- `recModel`: Path to recognition model file (default: `'rec.mnn'`)
- `dict`: Path to dictionary file (default: `'dict.txt'`)

**Returns:** `true` on successful initialization

### `detectText(imagePath: string): Promise<TextResult[]>`

Perform OCR on an image file.

**Returns:** Array of `TextResult` objects

### `detectTextFromBytes(imageData: string): Promise<TextResult[]>`

Perform OCR on base64-encoded image data.

**Returns:** Array of `TextResult` objects

### `detectTextToSpatialText(imagePath: string, yThresholdMultiplier?: number, xThresholdMultiplier?: number): Promise<string>`

Export OCR text formatted according to spatial position, with configurable line grouping (`yThresholdMultiplier`) and word/column spacing (`xThresholdMultiplier`).

### `getVersion(): Promise<string>`

Get the RustO library version.

### Types

```typescript
interface Frame {
  width: number;
  height: number;
  top: number;
  left: number;
}

interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  frame: Frame;
}

interface RustOConfig {
  detModelPath?: string;
  recModelPath?: string;
  dictPath?: string;
  clsModelPath?: string;
  orientModelPath?: string;
  unwarpModelPath?: string;
  orientThreshold?: number;
  clsThreshold?: number;
  textScore?: number;
  detThresh?: number;
  detBoxThresh?: number;
  limitSideLen?: number;
  limitType?: string;
  unclipRatio?: number;
  useDilation?: boolean;
  useDet?: boolean;
  useRec?: boolean;
  useCls?: boolean;
  useOrient?: boolean;
  useUnwarp?: boolean;
  debugImages?: boolean;
  minHeight?: number;
  maxSideLen?: number;
  minSideLen?: number;
  returnWordBox?: boolean;
  returnSingleCharBox?: boolean;
  yThresholdMultiplier?: number;
  xThresholdMultiplier?: number;
}
```

## Platform Support

- ✅ iOS 13.0+
- ✅ Android API 21+
- Architectures:
  - iOS: arm64, x86_64 (simulator)
  - Android: armeabi-v7a, arm64-v8a, x86, x86_64

## License

MIT
