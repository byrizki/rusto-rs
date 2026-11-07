import { WebOcrEngine, DetectTextBuilder } from './engine/webEngine.js';
import type {
  OcrRunOptions,
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
 * Checks if the default engine is ready for inference.
 */
export function isReady(): boolean {
  return defaultEngine.isReady();
}

/**
 * Checks if the default engine has been initialized.
 */
export function isInitialized(): boolean {
  return isReady();
}

/**
 * Returns the Web OCR engine library version string.
 */
export function version(): string {
  return defaultEngine.version();
}

/**
 * Returns the active inference engine backend identifier ('rusto-nn' | 'rusto-rten' | 'rusto-webgpu').
 */
export function engineBackend(): string {
  return defaultEngine.backend();
}

/**
 * Downloads or pre-fetches model asset URLs for a specific pipeline ahead of time.
 */
export function downloadModel(
  model?: string,
  lang?: string,
  cacheDir?: string
): Promise<string[]> {
  return defaultEngine.downloadModel(model, lang, cacheDir);
}


/**
 * Starts fluent builder chaining for OCR text detection.
 */
export function detectText(): DetectTextBuilder;
/**
 * Starts fluent builder chaining for OCR text detection with image source.
 */
export function detectText(source: ImageSource): DetectTextBuilder;
/**
 * Runs OCR text recognition on the given image source with spatial text output.
 */
export function detectText(
  source: ImageSource,
  options: OcrRunOptions & { output: 'spatial' }
): Promise<string>;
/**
 * Runs OCR text recognition on the given image source with structured lines or words output.
 */
export function detectText(
  source: ImageSource,
  options?: OcrRunOptions & { output?: 'lines' | 'words' }
): Promise<TextResult[]>;
/**
 * Runs OCR text recognition on the given image source.
 */
export function detectText(
  source: ImageSource,
  options?: OcrRunOptions
): Promise<TextResult[] | string>;
export function detectText(
  source?: ImageSource,
  options?: OcrRunOptions
): DetectTextBuilder | Promise<TextResult[] | string> {
  if (arguments.length === 0) {
    return defaultEngine.detectText();
  }
  if (arguments.length === 1 || options === undefined) {
    return defaultEngine.detectText(source!);
  }
  return defaultEngine.executeDetectText(source!, options);
}

/**
 * Runs OCR text recognition on raw interleaved 24-bit RGB pixel buffer with spatial text output.
 */
export function detectTextRgb(
  pixels: Uint8Array | ArrayBuffer,
  width: number,
  height: number,
  stride: number | null | undefined,
  options: OcrRunOptions & { output: 'spatial' }
): Promise<string>;
/**
 * Runs OCR text recognition on raw interleaved 24-bit RGB pixel buffer with structured lines or words output.
 */
export function detectTextRgb(
  pixels: Uint8Array | ArrayBuffer,
  width: number,
  height: number,
  stride?: number | null,
  options?: OcrRunOptions & { output?: 'lines' | 'words' }
): Promise<TextResult[]>;
/**
 * Runs OCR text recognition on raw interleaved 24-bit RGB pixel buffer.
 */
export function detectTextRgb(
  pixels: Uint8Array | ArrayBuffer,
  width: number,
  height: number,
  stride?: number | null,
  options?: OcrRunOptions
): Promise<TextResult[] | string>;
export function detectTextRgb(
  pixels: Uint8Array | ArrayBuffer,
  width: number,
  height: number,
  stride?: number | null,
  options?: OcrRunOptions
): Promise<TextResult[] | string> {
  return defaultEngine.detectTextRgb(pixels, width, height, stride, options);
}

/**
 * Releases worker resources and closes the default engine.
 */
export function close(): void {
  defaultEngine.close();
}

// Named exports
export { WebOcrEngine, DetectTextBuilder } from './engine/webEngine.js';
export * from './types/index.js';
export * from './utils/imageUtils.js';

