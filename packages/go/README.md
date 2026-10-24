# rusto (Go Binding)

High-performance, pure-Rust OCR engine bindings for Golang via Cgo.

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

## License

MIT License
