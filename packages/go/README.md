# RustO! Go SDK (`rusto`)

Official Go bindings (via Cgo) for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the pure-Rust MNN (`rusto-nn`) inference runtime.

## Features

- **Zero-Config Indra OCR**: `rusto.Initialize()` automatically downloads and caches the default `indraocr` models on first run from Hugging Face Hub.
- **Thread-Safe Native Handle**: Concurrent-safe `rusto.Engine` (`rusto.RustO`) synchronized with `sync.RWMutex` and deterministic `Close()`.
- **Document Intelligence**: Enable UVDoc unwarping (`WithAutoUnwarp(true)`), 4-way orientation detection (`WithOrientationDetection(true)`), and neural reading-order recovery (`WithLayoutReading(true)`).
- **Fluent Builder API**: Select `.AsLines()`, `.AsWords()`, `.AsSpatial()`, `.AsTsv()`, or `.WithOutput(...)` and execute with a single `.Run()` terminal call.
- **Multiple Input Modes**: File paths, in-memory PNG/JPEG `[]byte` slices, or raw 24-bit interleaved RGB buffers via `.RGB(...)`.

## Prerequisites & Installation

Build or install the native `librusto` shared library (`librusto.so`, `librusto.dylib`, or `rusto.dll`) from the repository root:

```bash
cargo build --release -p rusto-ffi
```

Add the Go package to your module:

```bash
go get github.com/byrizki/rusto-rs/packages/go
```

Ensure `librusto` is discoverable via your linker/runtime search path (for example `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS when running outside the monorepo).

## Quick Start (Zero-Config)

```go
package main

import (
	"fmt"
	"log"

	rusto "github.com/byrizki/rusto-rs/packages/go"
)

func main() {
	// Zero-config: initializes Indra OCR and auto-downloads models from Hugging Face Hub if missing
	engine, err := rusto.Initialize()
	if err != nil {
		log.Fatalf("failed to initialize RustO: %v", err)
	}
	defer engine.Close()

	// Structured line-level recognition with unwarping and layout reading order
	res, err := engine.
		DetectText("invoice.png").
		WithAutoUnwarp(true).
		WithLayoutReading(true).
		AsLines().
		Run()
	if err != nil {
		log.Fatalf("OCR failed: %v", err)
	}

	for _, item := range res.Results {
		fmt.Printf("[%.2f] (region=%d, layout=%d) %s\n",
			item.Score, item.RegionID, item.LayoutType, item.Text)
	}

	// Reconstructed 2D spatial layout
	spatialRes, err := engine.
		DetectText("invoice.png").
		AsSpatial().
		Run()
	if err != nil {
		log.Fatalf("spatial OCR failed: %v", err)
	}
	if spatialRes.IsSpatial {
		fmt.Println(spatialRes.Text)
	}
}
```

## Custom Configuration (`rusto.New` / `rusto.Initialize`)

Pass a JSON configuration string to `rusto.New(configJSON)` or `rusto.Initialize(configJSON)` to use specific presets (`"indra"`, `"ppv6"`, `"ppv5"`, `"ppv4"`) or offline `.mnn` / `.rten` files:

```go
configJSON := `{
  "template": "indra",
  "enableReadingOrder": true,
  "detection": {
    "enabled": true,
    "modelPath": "./models/indra_det.mnn"
  },
  "recognition": {
    "enabled": true,
    "modelPath": "./models/indra_rec.mnn",
    "dictPath": "./models/indra_meta.json"
  },
  "unwarp": {
    "enabled": true,
    "modelPath": "./models/uvdoc.mnn"
  },
  "layout": {
    "enabled": true,
    "modelPath": "./models/layout_cdla.mnn"
  }
}`

engine, err := rusto.New(configJSON)
if err != nil {
	log.Fatal(err)
}
defer engine.Close()
```

Or select a canonical model and language for automatic download:

```go
engine, err := rusto.New(`{"model": "ppocrv6-tiny", "lang": "latin"}`)
```

## Input Sources & Raw RGB Buffers

```go
// 1. Encoded image bytes (PNG, JPEG, WebP)
imgBytes, err := os.ReadFile("receipt.jpg")
if err != nil {
	log.Fatal(err)
}
wordsRes, err := engine.DetectText(imgBytes).AsWords().Run()

// 2. Raw interleaved 24-bit RGB pixel buffer
width, height := 1280, 720
stride := width * 3 // pass 0 to default to width * 3
rgbPixels := make([]byte, stride*height)

rgbRes, err := engine.
	DetectText().
	RGB(rgbPixels, width, height, stride).
	AsLines().
	Run()
```

## 6-Phase Pipeline Options

Chain per-request tuning stages on `DetectTextBuilder`:

```go
limitSide := 960
scoreThresh := float32(0.5)

res, err := engine.
	DetectText("scan.png").
	WithOrientationDetection(true).
	WithAutoUnwarp(true).
	WithLayoutReading(true).
	WithDetection(rusto.DetectionRunOptions{
		LimitSideLen: &limitSide,
		LimitType:    "max",
	}).
	WithPostRecognition(rusto.PostRecognitionOptions{
		ScoreThreshold: &scoreThresh,
	}).
	WithOutput(rusto.OutputLines).
	Run()
```

## Result Types

### `DetectTextResult`

| Field | Type | Description |
| :--- | :--- | :--- |
| `Results` | `[]TextResult` | Recognized lines or words when `IsSpatial` is `false` |
| `Text` | `string` | Formatted 2D spatial layout when `IsSpatial` is `true` |
| `IsSpatial` | `bool` | `true` when executed with `.AsSpatial()` or `.WithOutput(rusto.OutputSpatial)` |

### `TextResult`

| Field | Type | Description |
| :--- | :--- | :--- |
| `Text` | `string` | Recognized UTF-8 text |
| `Score` | `float32` | Recognition confidence (`0.0` – `1.0`) |
| `BoxPoints` | `[4][2]float32` | 4-vertex polygon `[top-left, top-right, bottom-right, bottom-left]` |
| `Frame` | `Frame` | Axis-aligned bounding box (`Width`, `Height`, `Top`, `Left`) |
| `RegionID` | `int32` | Layout region index when layout reading is enabled (`-1` if unassigned) |
| `LayoutType` | `int32` | Semantic layout category code (`-1` if unassigned; `0`=text, `1`=title, `4`=table, etc.) |

## License

Apache-2.0 OR MIT
