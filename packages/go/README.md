# rusto (Go Binding)

[![Go Reference](https://pkg.go.dev/badge/github.com/byrizki/rusto-rs/packages/go.svg)](https://pkg.go.dev/github.com/byrizki/rusto-rs/packages/go)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance, pure-Rust OCR engine bindings for Golang via Cgo.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

## Features

- **Native Cgo Bridge**: Direct in-memory C ABI bindings without subprocess overhead.
- **Embedded Runtime Rpath**: Built-in dynamic linker configuration (`-Wl,-rpath`) for seamless runtime resolution.
- **Structured & Spatial Results**: Line boxes, polygon coordinates, confidence scores, and physical text layouts.
- **Thread Safe**: Concurrent recognition support with standard Go finalizers.

## Installation

```bash
go get github.com/byrizki/rusto-rs/packages/go
```

Ensure `librusto.so` (Linux), `librusto.dylib` (macOS), or `librusto.dll` (Windows) is built and available.

## Quick Start

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/byrizki/rusto-rs/packages/go"
)

func main() {
	config := map[string]interface{}{
		"detection": map[string]string{
			"model_path": "models/PPOCR_v6/det.rten",
		},
		"recognition": map[string]string{
			"model_path": "models/PPOCR_v6/rec.rten",
			"dict_path":  "models/PPOCR_v6/dict.txt",
		},
	}
	configJSON, _ := json.Marshal(config)

	engine, err := rusto.New(string(configJSON))
	if err != nil {
		log.Fatalf("Failed to initialize RustO: %v", err)
	}
	defer engine.Close()

	// Detect structured lines
	results, err := engine.DetectTextFile("invoice.png", "{}")
	if err != nil {
		log.Fatalf("Detection failed: %v", err)
	}

	for _, item := range results {
		fmt.Printf("[%0.2f] %s\n", item.Score, item.Text)
	}

	// Detect spatial 2D formatted text
	spatial, err := engine.DetectTextFileSpatial("invoice.png", "{}")
	if err == nil {
		fmt.Println("--- Spatial Output ---")
		fmt.Println(spatial)
	}
}
```

## 📖 Documentation

For full documentation, guides, and API reference:
- [Golang Guide](https://byrizki.com/rusto-rs/how-to/use-golang)
- [Full Documentation](https://byrizki.com/rusto-rs/)

## License

MIT License

