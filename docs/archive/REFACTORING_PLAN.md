# RapidOCR Rust - Refactoring & Bindings Plan

## Status: **99.3% OpenCV Parity Achieved** ✅

## Phase 1: Code Cleanup ✅ IN PROGRESS

### 1.1 Remove Debug Code ✅ DONE
- ✅ Removed all `eprintln!` debug statements from `postprocess.rs`
- ✅ Removed pixel sampling debug code
- ✅ Removed foreground count logging

### 1.2 Fix Warnings
- ⏳ Remove unused functions in `contours.rs` (188+ lines of dead code)
- ⏳ Fix unused variable warnings (`start_x`, `start_y`, `nbd`)
- ⏳ Remove unused imports (`Luma` in tests only)
- ⏳ Add `#[allow(dead_code)]` for test utilities

### 1.3 Clean Cargo.toml
Dependencies to review:
- `geo-clipper`, `geo-types` - Used for polygon offsetting (unclip)
- `nalgebra` - Used for perspective transform (SVD)
- `imageproc` - Listed but not used? Check and remove
- Development dependencies cleanup

## Phase 2: Code Restructuring

### 2.1 Module Organization
```
src/
├── main.rs          # Single CLI entry point (replace bins/)
├── lib.rs           # Library API
├── core/
│   ├── mod.rs
│   ├── det.rs       # Detection
│   ├── rec.rs       # Recognition
│   └── cls.rs       # Classification
├── preprocess/
│   ├── mod.rs
│   └── image.rs
├── postprocess/
│   ├── mod.rs
│   ├── contours.rs
│   └── filtering.rs
├── geometry/
│   ├── mod.rs
│   ├── transform.rs
│   └── shapes.rs
└── ffi/             # FFI bindings
    ├── mod.rs
    ├── c.rs         # C API
    ├── csharp.rs    # C# helpers
    ├── android.rs   # JNI
    ├── ios.rs       # iOS
    └── react_native.rs # React Native JSI
```

### 2.2 API Design
```rust
// High-level API
pub struct RapidOCR {
    detector: TextDetector,
    recognizer: TextRecognizer,
}

impl RapidOCR {
    pub fn new(config: Config) -> Result<Self>;
    pub fn detect(&self, image: &[u8]) -> Result<Vec<TextBox>>;
    pub fn recognize(&self, image: &[u8], boxes: &[TextBox]) -> Result<Vec<TextResult>>;
    pub fn ocr(&self, image: &[u8]) -> Result<Vec<TextResult>>;
}
```

## Phase 3: Binary Consolidation

### 3.1 Remove Multiple Bins
Current bins to remove:
- `src/bin/rapidocr_json.rs`
- Any other bin files

### 3.2 Create Single main.rs
```rust
// src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run OCR on an image
    Ocr {
        /// Path to detection model
        #[arg(long)]
        det_model: PathBuf,
        /// Path to recognition model
        #[arg(long)]
        rec_model: PathBuf,
        /// Path to dictionary
        #[arg(long)]
        dict: PathBuf,
        /// Input image path
        image: PathBuf,
        /// Output format (json, text)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Benchmark performance
    Bench {
        // benchmark params
    },
}
```

## Phase 4: Release Optimization

### 4.1 Cargo.toml Profile
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false
```

### 4.2 Feature Flags
```toml
[features]
default = []
opencv = ["dep:opencv"]
simd = []  # Enable SIMD optimizations
```

### 4.3 Binary Size Optimization
- Remove unused dependencies
- Use `cargo bloat` to identify large dependencies
- Consider `wasm-opt` for smaller binaries

## Phase 5: C# Bindings

### 5.1 C API Layer
```rust
// src/ffi/c.rs
#[no_mangle]
pub extern "C" fn rapidocr_new(
    det_model: *const c_char,
    rec_model: *const c_char,
    dict: *const c_char,
) -> *mut RapidOCR { ... }

#[no_mangle]
pub extern "C" fn rapidocr_ocr(
    ocr: *mut RapidOCR,
    image_data: *const u8,
    image_len: usize,
    results_out: *mut *mut TextResult,
    count_out: *mut usize,
) -> i32 { ... }

#[no_mangle]
pub extern "C" fn rapidocr_free(ocr: *mut RapidOCR) { ... }
```

### 5.2 C# P/Invoke Wrapper
```csharp
// RapidOCR.NET/RapidOCR.cs
using System;
using System.Runtime.InteropServices;

namespace RapidOCR
{
    public class OCR : IDisposable
    {
        [DllImport("rapidocr", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rapidocr_new(string detModel, string recModel, string dict);
        
        [DllImport("rapidocr", CallingConvention = CallingConvention.Cdecl)]
        private static extern int rapidocr_ocr(IntPtr ocr, byte[] imageData, int imageLen, 
            out IntPtr results, out int count);
        
        private IntPtr handle;
        
        public OCR(string detModel, string recModel, string dict)
        {
            handle = rapidocr_new(detModel, recModel, dict);
        }
        
        public List<TextResult> Recognize(byte[] imageData) { ... }
        
        public void Dispose() { ... }
    }
}
```

### 5.3 NuGet Package
- Create `.nuspec` file
- Include native libraries for win-x64, linux-x64, osx-x64
- Auto-detect platform and load correct native library

## Phase 6: Android (AAR) Bindings

### 6.1 JNI Layer
```rust
// src/ffi/android.rs
#[cfg(target_os = "android")]
use jni::JNIEnv;
use jni::objects::{JClass, JString, JByteArray};
use jni::sys::{jlong, jint, jobjectArray};

#[no_mangle]
pub extern "system" fn Java_com_rapidocr_RapidOCR_nativeNew(
    env: JNIEnv,
    _class: JClass,
    det_model: JString,
    rec_model: JString,
    dict: JString,
) -> jlong {
    // Convert JStrings to Rust strings
    // Create RapidOCR instance
    // Return as jlong pointer
}

#[no_mangle]
pub extern "system" fn Java_com_rapidocr_RapidOCR_nativeOCR(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    image_data: JByteArray,
) -> jobjectArray {
    // Convert handle back to &mut RapidOCR
    // Process image
    // Convert results to Java objects
}
```

### 6.2 Kotlin Wrapper
```kotlin
// android/rapidocr/src/main/java/com/rapidocr/RapidOCR.kt
package com.rapidocr

class RapidOCR(
    detModel: String,
    recModel: String,
    dict: String
) {
    private val nativeHandle: Long
    
    init {
        System.loadLibrary("rapidocr")
        nativeHandle = nativeNew(detModel, recModel, dict)
    }
    
    fun recognize(imageData: ByteArray): List<TextResult> {
        return nativeOCR(nativeHandle, imageData).map { TextResult.fromNative(it) }
    }
    
    private external fun nativeNew(detModel: String, recModel: String, dict: String): Long
    private external fun nativeOCR(handle: Long, imageData: ByteArray): Array<Any>
    private external fun nativeFree(handle: Long)
    
    protected fun finalize() {
        nativeFree(nativeHandle)
    }
}
```

### 6.3 Gradle Build
```gradle
// android/rapidocr/build.gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'
        }
    }
}

dependencies {
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1'
}
```

## Phase 7: iOS (XCFramework) Bindings

### 7.1 C API with Swift-Friendly Types
```rust
// src/ffi/ios.rs
use std::os::raw::c_char;

#[repr(C)]
pub struct ROCRTextResult {
    pub text: *mut c_char,
    pub score: f32,
    pub box_points: [ROCRPoint; 4],
}

#[repr(C)]
pub struct ROCRPoint {
    pub x: f32,
    pub y: f32,
}

#[no_mangle]
pub extern "C" fn rocr_new(
    det_model: *const c_char,
    rec_model: *const c_char,
    dict: *const c_char,
) -> *mut c_void { ... }

#[no_mangle]
pub extern "C" fn rocr_ocr(
    handle: *mut c_void,
    image_data: *const u8,
    image_len: usize,
    results_out: *mut *mut ROCRTextResult,
    count_out: *mut usize,
) -> i32 { ... }
```

### 7.2 Swift Wrapper
```swift
// ios/RapidOCR/RapidOCR.swift
import Foundation

public class RapidOCR {
    private var handle: OpaquePointer?
    
    public init(detModel: String, recModel: String, dict: String) throws {
        handle = rocr_new(detModel, recModel, dict)
        guard handle != nil else {
            throw ROCRError.initializationFailed
        }
    }
    
    public func recognize(imageData: Data) throws -> [TextResult] {
        var results: UnsafeMutablePointer<ROCRTextResult>?
        var count: Int = 0
        
        let status = imageData.withUnsafeBytes { ptr in
            rocr_ocr(handle, ptr.baseAddress, ptr.count, &results, &count)
        }
        
        guard status == 0, let resultsPtr = results else {
            throw ROCRError.recognitionFailed
        }
        
        defer { rocr_free_results(resultsPtr, count) }
        
        return (0..<count).map { i in
            TextResult(from: resultsPtr[i])
        }
    }
    
    deinit {
        if let h = handle {
            rocr_free(h)
        }
    }
}
```

### 7.3 XCFramework Build Script
```bash
#!/bin/bash
# build_xcframework.sh

# Build for iOS device
cargo build --release --target aarch64-apple-ios
# Build for iOS simulator
cargo build --release --target aarch64-apple-ios-sim
cargo build --release --target x86_64-apple-ios

# Create XCFramework
xcodebuild -create-xcframework \
    -library target/aarch64-apple-ios/release/librapidocr.a \
    -headers ios/RapidOCR/include \
    -library target/aarch64-apple-ios-sim/release/librapidocr.a \
    -headers ios/RapidOCR/include \
    -output RapidOCR.xcframework
```

## Phase 8: React Native Bindings (JSI)

### 8.1 JSI C++ Bridge
```cpp
// react-native/cpp/RapidOCRJSI.h
#pragma once

#include <jsi/jsi.h>
#include <memory>

using namespace facebook;

class RapidOCRJSI : public jsi::HostObject {
public:
    RapidOCRJSI(
        jsi::Runtime& runtime,
        const std::string& detModel,
        const std::string& recModel,
        const std::string& dict
    );
    
    jsi::Value get(jsi::Runtime&, const jsi::PropNameID& name) override;
    void set(jsi::Runtime&, const jsi::PropNameID& name, const jsi::Value& value) override;
    std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& rt) override;
    
    jsi::Value recognize(jsi::Runtime& runtime, const jsi::Value& imageData);
    
private:
    void* rapidOCRHandle;
};
```

### 8.2 JSI Implementation
```cpp
// react-native/cpp/RapidOCRJSI.cpp
#include "RapidOCRJSI.h"
#include "rapidocr.h" // C API header

RapidOCRJSI::RapidOCRJSI(
    jsi::Runtime& runtime,
    const std::string& detModel,
    const std::string& recModel,
    const std::string& dict
) {
    rapidOCRHandle = rapidocr_new(detModel.c_str(), recModel.c_str(), dict.c_str());
}

jsi::Value RapidOCRJSI::recognize(jsi::Runtime& runtime, const jsi::Value& imageDataValue) {
    auto imageData = imageDataValue.asObject(runtime).getArrayBuffer(runtime);
    auto* data = imageData.data(runtime);
    auto size = imageData.size(runtime);
    
    TextResult* results;
    size_t count;
    
    int status = rapidocr_ocr(rapidOCRHandle, (uint8_t*)data, size, &results, &count);
    
    if (status != 0) {
        throw jsi::JSError(runtime, "OCR recognition failed");
    }
    
    // Convert results to JSI array
    jsi::Array jsResults(runtime, count);
    for (size_t i = 0; i < count; i++) {
        jsi::Object obj(runtime);
        obj.setProperty(runtime, "text", jsi::String::createFromUtf8(runtime, results[i].text));
        obj.setProperty(runtime, "score", jsi::Value(results[i].score));
        // ... set other properties
        jsResults.setValueAtIndex(runtime, i, obj);
    }
    
    rapidocr_free_results(results, count);
    return jsResults;
}
```

### 8.3 TypeScript Wrapper
```typescript
// react-native/src/index.ts
export interface TextResult {
  text: string;
  score: number;
  box: {
    topLeft: { x: number; y: number };
    topRight: { x: number; y: number };
    bottomRight: { x: number; y: number };
    bottomLeft: { x: number; y: number };
  };
}

export interface RapidOCRConfig {
  detModel: string;
  recModel: string;
  dict: string;
}

export class RapidOCR {
  private readonly nativeModule: any;

  constructor(config: RapidOCRConfig) {
    const { RapidOCRModule } = require('./NativeRapidOCR');
    this.nativeModule = new RapidOCRModule(
      config.detModel,
      config.recModel,
      config.dict
    );
  }

  async recognize(imageData: Uint8Array): Promise<TextResult[]> {
    return this.nativeModule.recognize(imageData);
  }
}
```

### 8.4 New Architecture Support
```typescript
// react-native/src/NativeRapidOCR.ts
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  recognize(imageData: Uint8Array): Promise<Array<Object>>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('RapidOCRModule');
```

## Implementation Priority

### Phase 1: Critical (Week 1)
1. ✅ Code cleanup (remove debug, fix warnings)
2. Clean Cargo.toml dependencies
3. Consolidate to single main.rs
4. Optimize release build

### Phase 2: Important (Week 2)
1. Restructure code for better organization
2. Create C API layer
3. C# bindings + NuGet package

### Phase 3: Mobile (Week 3-4)
1. Android AAR + Kotlin wrapper
2. iOS XCFramework + Swift wrapper

### Phase 4: React Native (Week 4-5)
1. JSI bridge implementation
2. TypeScript definitions
3. New Architecture support

## Testing Strategy

- Unit tests for each module
- Integration tests for OCR pipeline
- Platform-specific tests for each binding
- Performance benchmarks
- CI/CD with GitHub Actions

## Documentation

- API documentation (rustdoc)
- C# usage examples
- Android/Kotlin examples  
- iOS/Swift examples
- React Native examples
- Performance comparison guide
