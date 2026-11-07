# rusto-core 🦀

[![Crates.io](https://img.shields.io/crates/v/rusto-core.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-core)
[![docs.rs](https://img.shields.io/docsrs/rusto-core?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Core computer vision algorithms, postprocessing routines, preprocessing pipelines, and domain types for [RustO!](https://github.com/byrizki/rusto-rs) — the pure-Rust OCR engine and document intelligence toolkit.

`rusto-core` provides 100% pure-Rust, zero-C++ implementations of all preprocessing, geometric transformations, contour finding, unwarping, orientation correction, and text decoding required for state-of-the-art OCR.

---

## 🚀 Features

- **Zero-Overhead Tensor Container (`Tensor<T>`)** — Lightweight, contiguous row-major n-dimensional tensor container with multi-axis indexing (`[b, c, y, x]`), slicing, and axis transformations without external matrix dependencies.
- **Pure-Rust Text Detection Postprocessing** — Fast DBNet polygon contour extraction, unclipped box dilation, and polygonal minimum area bounding box calculation (`decode_dbnet_boxes_fast`).
- **CTC Decoder (`CtcDecoder`)** — Highly optimized Connectionist Temporal Classification decoding with greedy search, beam search, and character dictionary mapping for SVTR and CRNN text recognition models.
- **Pure-Rust 3D Document Unwarping (`UVDoc`)** — Complete 3D mesh deformation solver and backward grid sampling for curved, crumpled, or folded document rectification.
- **Document Layout & Reading Order Analysis** — PicoDet region classification (`regionId` / `layoutType`) and graph-based reading order sequencing.
- **Orientation & Calibration** — Neural 4-way rotation detection (0°, 90°, 180°, 270°) and automatic background descreening/equalization.
- **Extensible Inference Engine Trait (`InferenceEngine`)** — Abstract interface decoupling computer vision logic from underlying neural inference runtimes (`rusto-nn`, `rusto-rten`, `rusto-mnn`).

---

## 📦 Installation

Add `rusto-core` to your `Cargo.toml`:

```toml
[dependencies]
rusto-core = "0.4"
```

---

## 🛠️ Module Overview

| Module | Description |
|---|---|
| [`tensor`](src/tensor.rs) | Contiguous `Tensor<T>`, `Array2`, `Array3`, `Array4`, and `ArrayD` types |
| [`preprocess`](src/preprocess.rs) | Image resizing, padding, normalization, and affine crop extraction |
| [`det`](src/det.rs) | DBNet segmentation thresholding and polygon extraction |
| [`postprocess`](src/postprocess.rs) | Unclipping polygons and calculating oriented bounding boxes |
| [`ctc`](src/ctc.rs) | CTC greedy and beam-search text recognition decoding |
| [`unwarp`](src/unwarp.rs) | 3D document mesh unwarping and coordinate rectification |
| [`orient`](src/orient.rs) | 4-way document orientation classification and auto-rotation |
| [`layout`](src/layout.rs) | Layout region parsing and natural reading order sorting |
| [`spatial`](src/spatial.rs) | Spatial clustering into lines, paragraphs, and TSV export |
| [`engine`](src/engine.rs) | `InferenceEngine` trait and dynamic backend registry |
| [`pipeline`](src/pipeline.rs) | High-level 6-phase document OCR pipeline orchestrator |

---

## 💡 Example: Using the Tensor Abstraction

```rust
use rusto_core::tensor::Tensor;

// Create a 4D image tensor: [batch, channels, height, width]
let mut tensor = Tensor::<f32>::zeros(&[1, 3, 48, 320]);

// Access and modify elements
tensor[[0, 0, 10, 20]] = 1.0;

// View underlying contiguous memory buffer
let slice: &[f32] = tensor.as_slice();
assert_eq!(slice.len(), 1 * 3 * 48 * 320);
```

---

## 📄 License

Licensed under the [MIT License](../../LICENSE-MIT).

