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
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.0'     // ~6 MB (recommended default)
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-small:v0.2.0'    // ~30 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-medium:v0.2.0'   // ~134 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv5-mobile:v0.2.0'   // ~28 MB
    // or implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv4-mobile:v0.2.0'   // ~23 MB
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

// Initialize with unified grouped RustOConfig
await initialize({
  template: 'ppv6',
  detection: {
    modelPath: 'det.mnn',
    thresh: 0.3,
    boxThresh: 0.5,
  },
  recognition: {
    modelPath: 'rec.mnn',
    dictPath: 'dict.txt',
    scoreThresh: 0.6,
  },
  layout: {
    yThresholdMultiplier: 0.5,
    xThresholdMultiplier: 0.4,
  },
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

### `initialize(config?: RustOConfig): Promise<boolean>`

Initialize the RustO engine with optional `RustOConfig` configuration.

**Parameters:**
- `config`: Optional `RustOConfig` configuration object (supports template presets and specific property overrides).

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

interface DetectionConfig {
  enabled?: boolean;    // Enable/disable detection stage (default: true)
  modelPath?: string;   // Path to text detection model (e.g. 'det.mnn')
  thresh?: number;      // Binarization threshold (default: 0.3)
  boxThresh?: number;   // Box score threshold (default: 0.6 in v6/v4/v3, 0.5 in v5)
  unclipRatio?: number; // Polygon expansion ratio (default: 2.0 in v6/v5, 1.5 in v4/v3)
  limitSideLen?: number;// Max image side length limit (default: 736 in v6/v5, 960 in v4/v3)
  limitType?: string;   // Side limit type: 'min' | 'max'
  useDilation?: boolean;// Apply morphological dilation (default: true in v6/v5, false in v4/v3)
}

interface RecognitionConfig {
  enabled?: boolean;            // Enable/disable recognition stage (default: true)
  modelPath?: string;           // Path to text recognition model (e.g. 'rec.mnn')
  dictPath?: string;            // Path to dictionary file (e.g. 'dict.txt')
  scoreThresh?: number;         // Min text confidence score (default: 0.5)
  returnWordBox?: boolean;      // Return word-level boxes (default: false)
  returnSingleCharBox?: boolean;// Return character-level boxes (default: false)
}

/** Line Classification (CLS) — Available ONLY on PP-OCRv4 & PP-OCRv5 */
interface ClassificationConfig {
  enabled?: boolean;    // Enable line classifier (default: false, v4/v5 only)
  modelPath?: string;   // Path to line orientation model (v4/v5 only)
  thresh?: number;      // Min confidence threshold (default: 0.9, v4/v5 only)
}

interface OrientationConfig {
  enabled?: boolean;    // Enable document orientation correction (default: false)
  modelPath?: string;   // Path to document orientation model
  thresh?: number;      // Min orientation confidence threshold (default: 0.5)
}

interface UnwarpConfig {
  enabled?: boolean;    // Enable document unwarping (default: false)
  modelPath?: string;   // Path to document unwarping model
}

interface PreprocessingConfig {
  minHeight?: number;   // Minimum text box height in pixels (default: 30.0)
  maxSideLen?: number;  // Maximum image side length (default: 2000.0)
  minSideLen?: number;  // Minimum image side length (default: 30.0)
  debugImages?: boolean;// Return intermediate debug images (default: false)
}

interface LayoutConfig {
  yThresholdMultiplier?: number;// Y line grouping threshold (default: 0.5)
  xThresholdMultiplier?: number;// X word spacing threshold (default: 0.4)
}

interface RustOConfig {
  template?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3' | string;
  detection?: DetectionConfig;
  recognition?: RecognitionConfig;
  classification?: ClassificationConfig; // NOTE: Only in v4 & v5
  orientation?: OrientationConfig;
  unwarp?: UnwarpConfig;
  preprocessing?: PreprocessingConfig;
  layout?: LayoutConfig;
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
