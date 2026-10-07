import type { CalibrationOptions, ModelPreset, OptimizationOptions } from './options.js';

export interface ModelSourceConfig {
  detection?: string | Uint8Array | ArrayBuffer;
  recognition?: string | Uint8Array | ArrayBuffer;
  dictionary?: string | Uint8Array | ArrayBuffer;
  classification?: string | Uint8Array | ArrayBuffer;
  orientation?: string | Uint8Array | ArrayBuffer;
  [key: string]: string | Uint8Array | ArrayBuffer | undefined;
}

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

  /** Default pre-OCR image calibration. */
  calibration?: CalibrationOptions;

  /** Default runtime image optimization. */
  optimization?: OptimizationOptions;
}
