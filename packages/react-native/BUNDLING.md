# Model Provisioning & Offline Bundling (`@rustors/react-native`)

`@rustors/react-native` executes pure-Rust **`.rten`** OCR models natively on iOS and Android. You can choose between **zero-config automatic download** (smallest app binary) and **offline asset bundling** (zero network dependency on first launch).

> Deprecated v0.3 npm model wrapper packages and legacy model formats are no longer used in v0.4.0+. All pipelines run on standalone `.rten` models and vocabulary files.

---

## Strategy 1: Zero-Config On-Demand Download (Recommended Default)

Calling `await initialize()` with no file paths automatically downloads and caches the default **Indra OCR** (`indraocr`) `.rten` assets on first run:

- **Android cache location**: `<context.cacheDir>/rusto/models`
- **iOS cache location**: `<Library/Caches>/rusto/models`

Subsequent app launches load directly from the on-device cache with zero network traffic.

```ts
import { close, detectText, initialize } from '@rustors/react-native';

// Default high-accuracy Indra OCR pipeline
await initialize();

// Or select lightweight PP-OCRv6 Tiny for minimal download size (~8 MB total)
await initialize({
  model: 'ppocrv6-tiny',
  lang: 'latin',
});

const lines = await detectText({ uri: 'file:///path/to/receipt.jpg' })
  .asLines()
  .run();

await close();
```

---

## Strategy 2: Offline Bundling (`.mnn` / `.rten` + Dictionary Assets)

If your app must work completely offline from the very first launch (field inspection, point-of-sale, medical/government scanning), bundle standalone `.mnn` or `.rten` files directly in your native Android and iOS projects.

### 1. Download Standalone Assets

Download the `.mnn` (default, pure-Rust MNN runtime) or `.rten` models and dictionary for your chosen pipeline from [Hugging Face (`byrizki/rten-models`)](https://huggingface.co/byrizki/rten-models):

| Pipeline | Required Files | Optional Auxiliary Files |
| :--- | :--- | :--- |
| **Indra OCR (`preset: 'indra'`)** | `indra_det.mnn` (or `.rten`), `indra_rec.mnn` (or `.rten`), `indra_meta.json` | `uvdoc.mnn`, `indra_orient.mnn`, `layout_cdla.mnn`, `indra_order.mnn` |
| **PP-OCRv6 Tiny (`preset: 'ppv6'`)** | `ppocrv6_tiny_det.mnn` (or `.rten`), `ppocrv6_tiny_rec.mnn` (or `.rten`), `ppocrv6_dict.txt` | `uvdoc.mnn`, `ppocr_orient.mnn`, `ppocr_cls.mnn`, `layout_cdla.mnn` |

> **Tip (Convention-Based Zero-Config Offline)**: If you rename the three core files to `det.mnn` (or `det.rten`), `rec.mnn` (or `rec.rten`), and `dict.txt` (or `indra_meta.json`) and place them at the root of Android `assets/` and iOS Bundle Resources, calling `await initialize()` automatically detects and uses them offline without extra configuration!

### 2. Android Asset Placement (`assets/models/`)

Place your model and dictionary files under `android/app/src/main/assets/models/`:

```text
android/app/src/main/assets/
└── models/
    ├── ppocrv6_tiny_det.mnn
    ├── ppocrv6_tiny_rec.mnn
    ├── ppocrv6_dict.txt
    └── uvdoc.mnn
```

Optionally prevent Gradle from re-compressing `.mnn` and `.rten` binaries in `android/app/build.gradle`:

```groovy
android {
    androidResources {
        noCompress += ["mnn", "rten"]
    }
}
```

At runtime, `rusto-android` automatically extracts referenced APK assets into `context.cacheDir` on first initialization.

### 3. iOS Bundle Resource Placement

Add the model and dictionary files to your iOS target in Xcode:

1. Open `ios/<YourApp>.xcworkspace` in Xcode.
2. Drag your `.mnn` / `.rten` and `.txt` / `.json` files (or a `RustoModels.bundle` folder) into the project navigator.
3. Ensure **"Add to targets: `<YourApp>`"** is checked so they appear under **Build Phases -> Copy Bundle Resources**.

> Note: `RustO.resolveModelPath` on iOS resolves files placed in the main app bundle (`Bundle.main`), inside `RustoModels.bundle` / `RustOModels.bundle`, or in the app's `Documents` directory by filename.

### 4. Initialize & Run in JavaScript / TypeScript

When bundling inside `assets/models/` on Android and main bundle / `RustoModels.bundle` on iOS:

```ts
import { Platform } from 'react-native';
import { close, detectText, initialize } from '@rustors/react-native';

const assetPrefix = Platform.OS === 'android' ? 'models/' : '';

await initialize({
  preset: 'ppv6',
  models: {
    detection: `${assetPrefix}ppocrv6_tiny_det.mnn`,
    recognition: `${assetPrefix}ppocrv6_tiny_rec.mnn`,
    dictionary: `${assetPrefix}ppocrv6_dict.txt`,
    unwarp: `${assetPrefix}uvdoc.mnn`,
  },
});

// Structured line output
const results = await detectText({ uri: imageUri })
  .withAutoUnwarp()
  .asLines()
  .run();

// 2D spatial text layout
const spatialLayout = await detectText({ uri: imageUri })
  .asSpatial()
  .run();

await close();
```

*(If you place the files directly at the root of `android/app/src/main/assets/` on Android and Copy Bundle Resources on iOS, you can omit `assetPrefix` and pass identical filenames such as `'ppocrv6_tiny_det.mnn'` on both platforms.)*

---

## Strategy 3: Custom App-Managed Download Directory

If your app downloads `.mnn` or `.rten` models from your own authenticated backend (e.g., via `react-native-blob-util` or `expo-file-system`), pass absolute filesystem paths in `models` or `modelPath`:

```ts
await initialize({
  preset: 'indra',
  models: {
    detection: `${appSupportDir}/indra_det.mnn`,
    recognition: `${appSupportDir}/indra_rec.mnn`,
    dictionary: `${appSupportDir}/indra_meta.json`,
  },
});
```
```
