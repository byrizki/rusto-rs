# @rusto/react

> Official React hooks, provider, and UI components for [RustO!](https://github.com/byrizki/rusto-rs)

## Features

- **Provider Lifecycle**: Effortless engine initialization, status tracking (`idle`, `initializing`, `ready`, `error`), and error recovery with `<RustoProvider>`.
- **Easy Hooks**: `useRusto()` to access engine status, and `useOcr()` for running OCR and managing state (`isProcessing`, `results`, `spatialText`).
- **Composite Components**: `<OcrOverlay>` for displaying interactive bounding boxes, and `<OcrDropzone>` for drag-and-drop scanning.
- **Architectural Purity**: Functional logic is strictly separated into testable services and hooks; components remain clean and declarative.

## Installation

```bash
npm install @rusto/react @rusto/web
# or
yarn add @rusto/react @rusto/web
# or
pnpm add @rusto/react @rusto/web
```

## Quick Start

### 1. Wrap your application with `RustoProvider`

```tsx
import React from 'react';
import { RustoProvider } from '@rusto/react';
import { getModelUrls } from '@rusto/model-ppocrv6-tiny';

const modelUrls = getModelUrls('/models/ppocrv6-tiny');

export function App() {
  return (
    <RustoProvider
      config={{
        preset: 'ppv6',
        models: {
          detection: modelUrls.detection,
          recognition: modelUrls.recognition,
          dictionary: modelUrls.dictionary,
        },
      }}
    >
      <OcrScanner />
    </RustoProvider>
  );
}
```

### 2. Scan and Display with Hooks & Overlay

```tsx
import React, { useState } from 'react';
import { useOcr, OcrOverlay, OcrDropzone } from '@rusto/react';

export function OcrScanner() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const { detect, results, isProcessing } = useOcr();

  const handleFileSelect = async (file: File) => {
    setImageSrc(URL.createObjectURL(file));
    await detect(file);
  };

  return (
    <div>
      <OcrDropzone onFileSelect={handleFileSelect} />

      {isProcessing && <p>Scanning document...</p>}

      {imageSrc && results.length > 0 && (
        <OcrOverlay
          imageSrc={imageSrc}
          results={results}
          onItemClick={(item) => alert(`Clicked: ${item.text}`)}
        />
      )}
    </div>
  );
}
```

## License

MIT License
