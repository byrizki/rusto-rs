# React Native RustO

React Native OCR binding for RustO. Public API intentionally small:

```ts
await initialize();
const result = await detectText(source, options);
```

Initialize model resources once. Run `detectText` many times with request-local options.

## Install

```bash
yarn add react-native-rusto
# or npm install react-native-rusto
```

### iOS

Add matching model pod beside React Native pods:

```ruby
target 'YourApp' do
  pod 'RustO-Models-PPOCRv6-Tiny' # recommended default
  # pod 'RustO-Models-PPOCRv6-Small'
  # pod 'RustO-Models-PPOCRv6-Medium'
  # pod 'RustO-Models-PPOCRv5-Mobile'
  # pod 'RustO-Models-PPOCRv4-Mobile'
end
```

```bash
cd ios && pod install
```

`react-native-rusto`, `RustO`, and selected `RustO-Models-*` pod must use matching versions.

### Android

Enable JitPack, then add one matching model AAR:

```groovy
// android/settings.gradle or root build.gradle
dependencyResolutionManagement {
  repositories {
    google()
    mavenCentral()
    maven { url 'https://jitpack.io' }
  }
}

// android/app/build.gradle
dependencies {
  implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.5'
}
```

Use a matching model version. Custom absolute model paths need no model package.

## Complete example

```tsx
import { detectText, initialize, type TextResult } from 'react-native-rusto';

export async function readReceipt(uri: string): Promise<TextResult[]> {
  await initialize({ preset: 'ppv6' });

  return detectText(
    { uri },
    {
      output: 'words',
      textScore: 0.55,
      maxSideLen: 1600,
      detection: { limitSideLen: 960, limitType: 'max' },
      postprocess: { useDilation: true },
    }
  );
}

const words = await readReceipt('file:///absolute/path/receipt.jpg');
console.log(words);
```

Example result:

```json
[
  {
    "text": "Total",
    "score": 0.98,
    "box_points": [
      [40, 120],
      [124, 120],
      [124, 148],
      [40, 148]
    ],
    "frame": { "width": 84, "height": 28, "top": 120, "left": 40 }
  }
]
```

For approximate columns and rows:

```ts
const text = await detectText(
  { uri: 'file:///absolute/path/receipt.jpg' },
  { output: 'spatial', lineYThreshold: 0.5, wordXThreshold: 0.4 }
);

console.log(text);
// Item                 Amount
// Coffee                $3.50
// Total                $12.50
```

## Public API

### `initialize(config?: InitializeConfig): Promise<void>`

Creates or replaces native OCR engine. Defaults to PP-OCRv6 bundled model filenames.

```ts
await initialize({
  preset: 'ppv6',
  models: {
    detection: 'det.mnn',
    recognition: 'rec.mnn',
    dictionary: 'dict.txt',
    // classification: 'cls.mnn',
    // orientation: 'orient.mnn',
  },
});
```

| Config field            | Values                                 | Purpose                                                                 |
| ----------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `preset`                | `'ppv6'`, `'ppv5'`, `'ppv4'`, `'ppv3'` | Model-family defaults. Default: `'ppv6'`.                               |
| `models.detection`      | non-empty string                       | Detection-model path.                                                   |
| `models.recognition`    | non-empty string                       | Recognition-model path.                                                 |
| `models.dictionary`     | non-empty string                       | Recognition-dictionary path.                                            |
| `models.classification` | non-empty string                       | Optional classifier resource. Pair with request `classification: true`. |
| `models.orientation`    | non-empty string                       | Optional orientation resource. Pair with request `orientation: true`.   |

Initialization is model setup only. Resize, detection, and postprocess tuning belong to `detectText`.

### `detectText(source, options?)`

Runs one request. Native engine must already be initialized.

```ts
const lines = await detectText({ uri: '/absolute/path/invoice.jpg' });
const bytes = await detectText({ bytes: new Uint8Array(imageBytes) });
const base64 = await detectText({ base64: encodedJpeg });
```

`source` must contain **exactly one** of:

| Source       | Value                                   | Notes                                                                        |
| ------------ | --------------------------------------- | ---------------------------------------------------------------------------- |
| `{ uri }`    | non-empty string                        | Absolute path, `file:` URI, or Android `content://` URI. Whitespace trimmed. |
| `{ base64 }` | non-empty string                        | Encoded image payload. Whitespace trimmed.                                   |
| `{ bytes }`  | non-empty `Uint8Array` or `ArrayBuffer` | Encoded image bytes; bridge converts to base64.                              |

Unknown keys, multiple source keys, empty values, `null`, arrays, and primitive sources throw `TypeError` before native dispatch.

### `DetectTextOptions`

All fields optional. Omitted values merge with engine defaults and never alter later calls.

```ts
const options = {
  output: 'words' as const,
  lineYThreshold: 0.5,
  wordXThreshold: 0.4,
  textScore: 0.55,
  classification: false,
  orientation: false,
  minHeight: 32,
  maxSideLen: 1600,
  minSideLen: 64,
  widthHeightRatio: -1,
  detection: {
    limitSideLen: 960,
    limitType: 'max' as const,
    mean: [0.485, 0.456, 0.406] as [number, number, number],
    std: [0.229, 0.224, 0.225] as [number, number, number],
  },
  postprocess: {
    threshold: 0.3,
    boxThreshold: 0.6,
    maxCandidates: 1000,
    unclipRatio: 2.0,
    useDilation: true,
  },
};
```

| Field                                   | Valid value                       | Default / behavior                                                          |
| --------------------------------------- | --------------------------------- | --------------------------------------------------------------------------- |
| `output`                                | `'lines' \| 'words' \| 'spatial'` | `'lines'`; structured list for lines/words, string for spatial.             |
| `lineYThreshold`, `wordXThreshold`      | finite `>= 0`                     | `0.5`, `0.4`; line/word grouping tolerance.                                 |
| `textScore`                             | finite `[0, 1]`                   | Engine default confidence cutoff.                                           |
| `classification`, `orientation`         | boolean                           | `false`; only useful with configured optional models.                       |
| `minHeight`, `maxSideLen`, `minSideLen` | finite `> 0`                      | Request-local resize bounds; `minSideLen <= maxSideLen` when both supplied. |
| `widthHeightRatio`                      | finite `> 0` or `-1`              | `-1` retains native aspect-ratio behavior.                                  |
| `detection.limitSideLen`                | integer `1..32767`                | Detector resize bound.                                                      |
| `detection.limitType`                   | `'min' \| 'max'`                  | Short-side minimum / long-side maximum mode.                                |
| `detection.mean`, `detection.std`       | exactly three finite numbers      | Input normalization; `std` entries cannot be zero.                          |
| `postprocess.threshold`, `boxThreshold` | finite `[0, 1]`                   | Detector pixel and polygon confidence thresholds.                           |
| `postprocess.maxCandidates`             | integer `>= 1`                    | Candidate cap.                                                              |
| `postprocess.unclipRatio`               | finite `> 0`                      | Polygon expansion ratio.                                                    |
| `postprocess.useDilation`               | boolean                           | Enables dilation for faint/broken strokes.                                  |

`preprocessing` is not a valid nested option. `detection` and `postprocess` are root-level siblings.

## Result types

```ts
interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  frame: { width: number; height: number; top: number; left: number };
}
```

`box_points` order is top-left, top-right, bottom-right, bottom-left. Values are image pixels with origin at top-left. `frame` is axis-aligned, including for rotated polygons.

## Errors and platform support

- Call `initialize` before `detectText`; native bridge reports not-initialized failure otherwise.
- Invalid config/options/source fail before native OCR. Correct value and retry; initialized engine remains usable.
- Native image/model errors depend on platform: missing file, missing model resources, decode failure, or OCR runtime failure.
- Requires iOS 13+ or Android API 21+. Expo Go cannot load native Rust module; use development build.

## Model bundling

See [BUNDLING.md](./BUNDLING.md) for custom model packaging, on-demand resources, and app-size tradeoffs.

## License

MIT
