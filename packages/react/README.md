# @rustors/react

Official React hooks, context provider, and UI components for **RustO!** — client-side WebAssembly OCR powered by [`@rustors/web`](../web/README.md), Indra OCR, and PaddleOCR.

## Features

- **Zero-Backend Privacy**: Runs OCR in the browser via WebAssembly + Web Workers (`@rustors/web`).
- **Standalone `.rten` Model URLs**: Load `.rten` models directly from `public/models/`, your CDN, Hugging Face, or GitHub Releases — no npm model wrapper packages required.
- **Indra OCR & PP-OCRv6 Presets**: Supports `'indra'` alongside `'ppv6'`, `'ppv5'`, `'ppv4'`, and `'ppv3'`, plus optional `unwarp`, `orientation`, `layout`, and `order` `.rten` models.
- **Hooks & Ready-Made Components**: `<RustoProvider>`, `useOcr()`, `useRusto()`, `<OcrDropzone>`, `<OcrOverlay>`, and direct fluent `DetectTextBuilder` access (`.asLines().run()`, `.asWords().run()`, `.asSpatial().run()`).

## Installation

```bash
npm install @rustors/react @rustors/web
# or
pnpm add @rustors/react @rustors/web
```

## Configuring `.rten` Models with `<RustoProvider>`

Point `models` to standalone `.rten` URLs hosted in your app's `public/models/` folder, a CDN, Hugging Face (`https://huggingface.co/byrizki/rten-models/resolve/main`), or GitHub Releases (`https://github.com/byrizki/rusto-rs/releases/download/models-rten-v0.4.0`):

```tsx
import { RustoProvider } from '@rustors/react';

const HF_BASE = 'https://huggingface.co/byrizki/rten-models/resolve/main';

export function App() {
  return (
    <RustoProvider
      config={{
        preset: 'indra', // or 'ppv6' for lightweight PP-OCRv6 Tiny
        enableReadingOrder: true,
        models: {
          detection: `${HF_BASE}/indra_det.rten`,
          recognition: `${HF_BASE}/indra_rec.rten`,
          dictionary: `${HF_BASE}/indra_meta.json`,
          // Optional auxiliary document models:
          unwarp: `${HF_BASE}/uvdoc.rten`,
          orientation: `${HF_BASE}/indra_orient.rten`,
          layout: `${HF_BASE}/layout_cdla.rten`,
          order: `${HF_BASE}/indra_order.rten`,
        },
      }}
    >
      <InvoiceScanner />
    </RustoProvider>
  );
}
```

## Using the `useOcr` Hook & UI Components

```tsx
import { useState } from 'react';
import { OcrDropzone, OcrOverlay, useOcr, useRusto } from '@rustors/react';

export function InvoiceScanner() {
  const { status } = useRusto();
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const { detect, results, spatialText, isProcessing, error } = useOcr({
    autoUnwarp: true,
    layoutReading: true,
    output: 'lines',
  });

  const handleFile = async (file: File) => {
    setImageUrl(URL.createObjectURL(file));
    await detect(file);
  };

  if (status !== 'ready') {
    return <p>Loading OCR engine ({status})...</p>;
  }

  return (
    <div>
      <OcrDropzone onFileAccepted={handleFile} disabled={isProcessing} />
      {error && <p role="alert">{error.message}</p>}

      {imageUrl && (
        <div style={{ position: 'relative', display: 'inline-block' }}>
          <img src={imageUrl} alt="Uploaded document" style={{ maxWidth: '100%' }} />
          <OcrOverlay results={results} />
        </div>
      )}

      <ul>
        {results.map((item, idx) => (
          <li key={idx}>
            [{item.score.toFixed(2)}] ({item.layout_type ?? 'line'} #{item.region_id ?? '-'}){' '}
            {item.text}
          </li>
        ))}
      </ul>

      {spatialText && <pre>{spatialText}</pre>}
    </div>
  );
}
```

## Direct Fluent Builder Access (`asLines` / `asWords` / `asSpatial`)

When you want chained fluent builder methods directly inside a component or custom hook, access the underlying `WebOcrEngine` via `useRusto()` (or import `detectText` from `@rustors/web`):

```tsx
import { useRusto } from '@rustors/react';

export function useFluentScanner() {
  const { service } = useRusto();

  const scanInvoice = async (file: File) => {
    const engine = service.getEngine();

    // Line-level structured results with document unwarping & layout reading order
    const lines = await engine
      .detectText(file)
      .withAutoUnwarp()
      .withLayoutReading()
      .asLines()
      .run();

    // 2D spatial plain-text layout
    const spatial = await engine
      .detectText(file)
      .asSpatial()
      .run();

    return { lines, spatial };
  };

  return { scanInvoice };
}
```

## Exported Utilities

- **Hooks & Context**: `RustoProvider`, `RustoContext`, `useRusto`, `useOcr`
- **Components**: `OcrDropzone`, `DropzonePrompt`, `DropzonePreview`, `OcrOverlay`, `BoundingBox`
- **Formatters & Geometry**: `resultsToText`, `resultsToCsv`, `resultsToJson`, `scaleFrame`, `scaleQuad`, `calculateScale`

## License

Apache-2.0 OR MIT
