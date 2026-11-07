# rusto-cpp 🔬

[![Crates.io](https://img.shields.io/crates/v/rusto-cpp.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-cpp)
[![docs.rs](https://img.shields.io/docsrs/rusto-cpp?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optional native C++ and OpenCV accelerated computer vision backends for [RustO!](https://github.com/byrizki/rusto-rs).

`rusto-cpp` provides legacy C++ and OpenCV bindings for geometric transformations, polygon clipping (Clipper2), and contour extraction. It serves as an optional fallback and ground-truth validation suite against RustO!'s default 100% pure-Rust computer vision algorithms (`rusto-core`).

---

## 🚀 Features

- **OpenCV Interoperability** — Native OpenCV image decoding, color conversion, and contour detection via `opencv` crate.
- **Clipper2 Polygon Offsetting** — Robust polygon dilation and unclipping via native Clipper2 bindings.
- **Ground-Truth Verification** — Used by test suites to verify exact numerical and visual parity with pure-Rust algorithms.

---

## 📦 Requirements

When building with the `opencv` feature:
- OpenCV 4.x installed on the system (e.g. `libopencv-dev` on Debian/Ubuntu or `opencv` on macOS via Homebrew)
- Clang / LLVM for generating Rust bindings via `opencv-rust`

---

## 📦 Installation

Add `rusto-cpp` to your `Cargo.toml`:

```toml
[dependencies]
rusto-cpp = { version = "0.4", features = ["opencv"] }
```

---

## 📄 License

Licensed under the [MIT License](../../LICENSE-MIT).

