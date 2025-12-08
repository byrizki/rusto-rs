# Bundling Model Files with React Native RustO

This guide explains how to bundle OCR model files with your React Native application for both Android and iOS platforms.

## Overview

The React Native RustO package supports bundling model files directly into your application, eliminating the need to download models at runtime or manually specify file paths.

## Default Model Files

The package expects these default model files:
- `det.mnn` - Detection model
- `rec.mnn` - Recognition model  
- `dict.txt` - Character dictionary

These files should be placed in the appropriate directories for each platform.

---

## Android Setup

### Option 1: Bundle with Main Android Package

Place model files in `packages/android/src/main/assets/`:

```
packages/android/src/main/assets/
├── det.mnn
├── rec.mnn
└── dict.txt
```

The React Native package will automatically include these assets through the dependency link.

### Option 2: Bundle with React Native Package

Place model files in `packages/react-native/android/src/main/assets/`:

```
packages/react-native/android/src/main/assets/
├── det.mnn
├── rec.mnn
└── dict.txt
```

### How It Works

The Android `build.gradle` is configured to include assets from both locations:

```gradle
sourceSets {
  main {
    assets.srcDirs = [
      'src/main/assets',
      '../../../android/src/main/assets'
    ]
  }
}
```

When the app runs, the `RustO.create()` method automatically extracts these assets from the APK to the app's cache directory.

---

## iOS Setup

### 1. Create Models Directory

Create a `models` directory inside `packages/react-native/ios/`:

```bash
mkdir -p packages/react-native/ios/models
```

### 2. Copy Model Files

Place your model files in this directory:

```
packages/react-native/ios/models/
├── det.mnn
├── rec.mnn
└── dict.txt
```

### 3. Bundle Configuration

The `.podspec` file is already configured to bundle these models as resources:

```ruby
s.resource_bundles = {
  'RustoModels' => [
    'ios/models/*.mnn',
    'ios/models/*.txt'
  ]
}
```

### How It Works

When the app runs, the Swift module checks the `RustoModels.bundle` resource bundle first, then falls back to the main bundle and documents directory.

---

## Usage

### With Bundled Models (Recommended)

Simply call `initialize()` without parameters to use the bundled default models:

```typescript
import { initialize, detectText } from 'react-native-rusto';

// Initialize with bundled models
await initialize();

// Perform OCR
const results = await detectText('/path/to/image.jpg');
```

### With Custom Model Paths

You can still specify custom model paths if needed:

```typescript
// Initialize with custom models
await initialize(
  '/custom/path/det_model.mnn',
  '/custom/path/rec_model.mnn',
  '/custom/path/dictionary.txt'
);
```

### Partial Override

You can override only specific models:

```typescript
// Use bundled det.mnn and rec.mnn, but custom dictionary
await initialize(null, null, '/custom/path/my_dict.txt');

// Or use default for some
await initialize('/custom/det.mnn'); // Uses bundled rec.mnn and dict.txt
```

---

## File Size Considerations

Model files can be large (typically 5-50 MB each). Consider:

1. **APK/IPA Size**: Bundled models increase your app's download size
2. **Compression**: Both Android and iOS compress assets, reducing the impact
3. **On-Demand Loading**: For smaller app sizes, download models on first launch instead

---

## Copying Models from Source

To copy the default PPOCR_v5 models from the source repository:

### Android

```bash
# From the repository root
cp models/PPOCR_v5/det.mnn packages/android/src/main/assets/
cp models/PPOCR_v5/rec.mnn packages/android/src/main/assets/  
cp models/PPOCR_v5/dict.txt packages/android/src/main/assets/
```

### iOS

```bash
# From the repository root
mkdir -p packages/react-native/ios/models
cp models/PPOCR_v5/det.mnn packages/react-native/ios/models/
cp models/PPOCR_v5/rec.mnn packages/react-native/ios/models/
cp models/PPOCR_v5/dict.txt packages/react-native/ios/models/
```

---

## Verification

### Android

Build and inspect the AAR:

```bash
cd packages/react-native/android
./gradlew assembleRelease

# Inspect the AAR contents
unzip -l build/outputs/aar/android-release.aar | grep assets
```

You should see:
```
assets/det.mnn
assets/rec.mnn
assets/dict.txt
```

### iOS

Build and inspect the app:

```bash
cd ios
pod install
```

The models will be bundled in `RustoModels.bundle` within your app's frameworks.

---

## Troubleshooting

### Models Not Found (Android)

1. Verify files are in `src/main/assets/`
2. Clean and rebuild: `./gradlew clean assembleRelease`
3. Check LogCat for "Failed to find model files" errors

### Models Not Found (iOS)

1. Verify files are in `ios/models/`
2. Run `pod install` to update the bundle
3. Clean build folder in Xcode (Cmd+Shift+K)
4. Check Xcode logs for resource bundle errors

### Large App Size

If bundled models make your app too large, consider:
1. Using smaller model variants
2. Downloading models on first launch
3. Using on-demand resources (iOS) or app bundles (Android)
