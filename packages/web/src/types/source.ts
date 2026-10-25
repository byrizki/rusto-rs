/**
 * Binary image payload represented as Uint8Array or ArrayBuffer.
 */
export type BinaryImageData = Uint8Array | ArrayBuffer;

/**
 * Normalized bridge object for cross-platform image inputs.
 */
export type ImageSourceBridgeObject =
  | { uri: string }
  | { base64: string }
  | { bytes: BinaryImageData };

/**
 * Universal image source supported by the Web OCR engine:
 * - URL or Data URI string
 * - Blob / File
 * - ImageData
 * - ImageBitmap
 * - HTMLImageElement
 * - HTMLCanvasElement
 * - Raw binary bytes (Uint8Array / ArrayBuffer)
 * - Structured bridge object
 */
export type ImageSource =
  | string
  | Blob
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | BinaryImageData
  | ImageSourceBridgeObject;
