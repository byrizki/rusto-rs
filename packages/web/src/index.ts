import { WebOcrEngine } from './engine/webEngine.js';
import type {
  DetectTextOptions,
  InitializeConfig,
  ImageSource,
  TextResult,
} from './types/index.js';

// Default singleton instance
const defaultEngine = new WebOcrEngine();

/**
 * Initializes the default RustO Web OCR engine.
 */
export function initialize(config?: InitializeConfig): Promise<void> {
  return defaultEngine.initialize(config);
}

/**
 * Checks if the default engine has been initialized.
 */
export function isInitialized(): boolean {
  return defaultEngine.isInitialized();
}

/**
 * Runs OCR text recognition on the given image source with spatial text output.
 */
export function detectText(
  source: ImageSource,
  options: DetectTextOptions & { output: 'spatial' }
): Promise<string>;

/**
 * Runs OCR text recognition on the given image source with structured lines or words output.
 */
export function detectText(
  source: ImageSource,
  options?: DetectTextOptions & { output?: 'lines' | 'words' }
): Promise<TextResult[]>;

/**
 * Runs OCR text recognition on the given image source.
 */
export function detectText(
  source: ImageSource,
  options?: DetectTextOptions
): Promise<TextResult[] | string>;

export function detectText(
  source: ImageSource,
  options: DetectTextOptions = {}
): Promise<TextResult[] | string> {
  return defaultEngine.detectText(source, options as any);
}

/**
 * Terminates workers and resets default engine state.
 */
export function terminate(): void {
  defaultEngine.terminate();
}

// Named exports
export { WebOcrEngine } from './engine/webEngine.js';
export * from './types/index.js';
export * from './utils/geometry.js';
export * from './utils/imageUtils.js';
export * from './utils/validator.js';
