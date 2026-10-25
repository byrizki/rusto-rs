import type { InitializeConfig } from '@rustors/web';
import type { OcrService } from '../services/ocrService.js';

/**
 * Status lifecycle states for the RustO engine provider:
 * - 'idle': Not yet initialized.
 * - 'initializing': Loading WASM runtime and model weights.
 * - 'ready': Engine is ready to perform OCR detections.
 * - 'error': Engine initialization failed.
 */
export type RustoStatus = 'idle' | 'initializing' | 'ready' | 'error';

/**
 * Context value provided by `<RustoProvider>`.
 */
export interface RustoContextValue {
  /** Current lifecycle status of the engine. */
  status: RustoStatus;
  /** Boolean flag indicating whether status is 'ready'. */
  isReady: boolean;
  /** Initialization error if status is 'error'. */
  error: Error | null;
  /** Manually trigger engine initialization with configuration. */
  initialize: (config?: InitializeConfig) => Promise<void>;
  /** Underlying OCR service instance. */
  service: OcrService;
}

/**
 * Props for `<RustoProvider>`.
 */
export interface RustoProviderProps {
  /** Child React tree. */
  children: React.ReactNode;
  /** Engine initialization configuration. */
  config?: InitializeConfig;
  /** Whether to automatically initialize on component mount. Default: true. */
  autoInitialize?: boolean;
}
