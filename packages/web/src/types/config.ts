import type { CalibrationOptions, ModelPreset, OptimizationOptions } from './options.js';

/**
 * Model resource URLs or binary byte buffers for initialization.
 */
export interface ModelSourceConfig {
  /** Detection model URL (.rten or .onnx) or binary buffer. */
  detection?: string | Uint8Array | ArrayBuffer;
  /** Recognition model URL (.rten or .onnx) or binary buffer. */
  recognition?: string | Uint8Array | ArrayBuffer;
  /** Character dictionary vocabulary text URL (.txt) or binary buffer. */
  dictionary?: string | Uint8Array | ArrayBuffer;
  /** Optional text-line orientation classifier model URL or buffer. */
  classification?: string | Uint8Array | ArrayBuffer;
  /** Optional document orientation angle classifier model URL or buffer. */
  orientation?: string | Uint8Array | ArrayBuffer;
  [key: string]: string | Uint8Array | ArrayBuffer | undefined;
}

/**
 * Supported pluggable WebAssembly inference engine backends:
 * - 'rusto-nn': 100% pure-Rust Alibaba MNN/ONNX neural network engine (default, recommended).
 * - 'rusto-rten': RTen pure-Rust neural network inference engine.
 * - 'rusto-webgpu': WebGPU hardware accelerated compute shader engine (rusto-nn-webgpu).
 */
export type InferenceBackend = 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';

/**
 * Configuration options for initializing the RustO Web OCR engine.
 */
export interface InitializeConfig {
  /** Inference engine backend switch ('rusto-nn' | 'rusto-rten' | 'rusto-webgpu'). Defaults to 'rusto-nn'. */
  backend?: InferenceBackend;

  /** Model preset identifier. Defaults to 'ppv6'. */
  preset?: ModelPreset;

  /** Custom model asset URLs or binary buffers. */
  models?: ModelSourceConfig;

  /** Custom WebAssembly runtime binary URL or buffer. */
  wasmUrl?: string | ArrayBuffer | Uint8Array;

  /** Run OCR in a dedicated Web Worker (default: true in browser). */
  worker?: boolean;

  /** Custom Web Worker script URL. */
  workerUrl?: string;

  /** Number of compute threads if WebAssembly multithreading is enabled. */
  numThreads?: number;

  /** Enable multi-column reading order recovery sequence. */
  enableReadingOrder?: boolean;

  /** Layout / reading order architecture variant or model identifier (e.g. 'indra', 'cdla', 'publaynet', 'heuristic', 'none'). */
  layoutModel?: string;

  /** Default pre-OCR image calibration options. */
  calibration?: CalibrationOptions;

  /** Default runtime image optimization options. */
  optimization?: OptimizationOptions;

  /** Enable WebGPU hardware acceleration if supported by browser/runtime. Default: false. */
  webgpu?: boolean;
}
