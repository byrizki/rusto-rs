export type BinaryImageData = Uint8Array | ArrayBuffer;

export type ImageSourceBridgeObject =
  | { uri: string }
  | { base64: string }
  | { bytes: BinaryImageData };

export type ImageSource =
  | string
  | Blob
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | BinaryImageData
  | ImageSourceBridgeObject;
