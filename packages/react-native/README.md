# React Native RustO

React Native bindings for the RustO! OCR library.

## Installation

```bash
npm install react-native-rusto
# or
yarn add react-native-rusto
```

### iOS Setup

```bash
cd ios && pod install
```

Make sure to place the `RustO.xcframework` in the `ios/` directory.

### Android Setup

The Android native library will be linked automatically. Ensure that:

1. The main `rusto-android` package is available in your project
2. Your project includes the native `.so` libraries in the correct architecture folders

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
  console.log(`Box: ${result.box_points}`);
});

// Get library version
const version = await getVersion();
console.log(`RustO version: ${version}`);
```

### Advanced Usage (with custom models)

```typescript
import { initialize, detectText } from 'react-native-rusto';

// Initialize with custom model files
await initialize(
  '/custom/path/det_model.mnn',    // Detection model
  '/custom/path/rec_model.mnn',    // Recognition model
  '/custom/path/dict.txt'          // Dictionary file
);

// Or partially override (use bundled models for unspecified parameters)
await initialize(undefined, undefined, '/custom/dict.txt');
```

### Model Bundling

Model files can be bundled with your app. See [BUNDLING.md](./BUNDLING.md) for detailed instructions on how to:
- Bundle models with Android AAR
- Bundle models with iOS XCFramework
- Reduce app size with on-demand loading

## API

### `initialize(detModel?: string, recModel?: string, dict?: string): Promise<boolean>`

Initialize the RustO engine with model files.

**Parameters (all optional):**
- `detModel`: Path to detection model file (default: `'det.mnn'` from bundled assets)
- `recModel`: Path to recognition model file (default: `'rec.mnn'` from bundled assets)
- `dict`: Path to dictionary file (default: `'dict.txt'` from bundled assets)

Model files are searched in this order:
1. **iOS**: Resource bundle → Main bundle → Documents directory
2. **Android**: Custom path → Assets (auto-extracted to cache)

**Returns:** `true` on successful initialization

**Examples:**
```typescript
// Use bundled models (recommended)
await initialize();

// Use custom models
await initialize('/path/to/det.mnn', '/path/to/rec.mnn', '/path/to/dict.txt');

// Mix bundled and custom
await initialize(undefined, undefined, '/custom/dict.txt');
```

### `detectText(imagePath: string): Promise<TextResult[]>`

Perform OCR on an image file.

**Parameters:**
- `imagePath`: Absolute path to the image file

**Returns:** Array of `TextResult` objects

### `detectTextFromBytes(imageData: string): Promise<TextResult[]>`

Perform OCR on base64-encoded image data.

**Parameters:**
- `imageData`: Base64-encoded image data

**Returns:** Array of `TextResult` objects

### `getVersion(): Promise<string>`

Get the RustO library version.

**Returns:** Version string

### TextResult

```typescript
interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
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
