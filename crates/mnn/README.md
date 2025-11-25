# MNN Rust Bindings

Safe Rust bindings for the MNN (Mobile Neural Network) inference library.

## Overview

This crate provides a high-level, safe Rust interface to MNN, built on top of `mnn-sys` which provides the low-level FFI bindings.

## Features

- **Safe API**: All unsafe operations are wrapped in safe Rust abstractions
- **Zero-copy operations**: Efficient tensor data transfer
- **Platform support**: CPU, Metal (macOS/iOS), CoreML, OpenCL, CUDA, and more
- **Configurable backends**: Choose the best backend for your platform

## Usage

```rust
use mnn::{Interpreter, ScheduleConfig, ForwardType, BackendConfig, PrecisionMode, PowerMode};
use std::path::Path;

// Load model
let interpreter = Interpreter::from_file(Path::new("model.mnn"))?;

// Configure session
let mut config = ScheduleConfig::new();
config.set_type(ForwardType::Auto);

let mut backend_config = BackendConfig::new();
backend_config.set_precision_mode(PrecisionMode::High);
backend_config.set_power_mode(PowerMode::High);
config.set_backend_config(backend_config);

// Create session
let mut session = interpreter.create_session(config)?;

// Get input tensor
let mut input_tensor = interpreter.input(&mut session, "input")?;

// ... prepare input data ...

// Run inference
interpreter.run_session(&mut session)?;

// Get output
let output_tensor = interpreter.output(&session, "output")?;
```

## Platform-Specific Features

### macOS/iOS

Enable Metal and CoreML backends:

```toml
[dependencies]
mnn = { path = "crates/mnn", features = ["metal", "coreml"] }
```

### Linux/Android

OpenCL backend is enabled by default on these platforms.

## Building

This crate requires:
- CMake 3.10+
- C++11 compiler
- Platform-specific SDK (e.g., Metal SDK for macOS)

The MNN library is compiled from source during the build process.
