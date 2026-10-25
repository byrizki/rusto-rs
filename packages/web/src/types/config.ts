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
 * Configuration options for initializing the RustO Web OCR engine.
 */
export interface InitializeConfig {
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

  /** Default pre-OCR image calibration options. */
  calibration?: CalibrationOptions;

  /** Default runtime image optimization options. */
  optimization?: OptimizationOptions;
}
