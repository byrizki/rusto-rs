/**
 * Axis-aligned bounding box frame with position and size.
 */
export interface Frame {
  /** Width of the bounding box in pixels. */
  width: number;
  /** Height of the bounding box in pixels. */
  height: number;
  /** Top vertical coordinate (Y) of the bounding box. */
  top: number;
  /** Left horizontal coordinate (X) of the bounding box. */
  left: number;
}

/**
 * Recognized OCR text result containing text, confidence score, bounding polygon, and frame.
 */
export interface TextResult {
  /** Recognized text string. */
  text: string;
  /** Confidence score between 0.0 and 1.0. */
  score: number;
  /** Quad points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] representing polygon vertices. */
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  /** Axis-aligned bounding box frame. */
  frame: Frame;
}

/**
 * Image detection stage options.
 */
export interface DetectionOptions {
  /** Target side length constraint for image resizing. */
  limitSideLen?: number;
  /** Side length limit behavior: 'min' ensures short side meets threshold, 'max' caps long side. */
  limitType?: 'min' | 'max';
  /** Mean RGB normalization values. */
  mean?: [number, number, number];
  /** Standard deviation RGB normalization values. */
  std?: [number, number, number];
}

/**
 * Post-processing options for DBNet polygon detection and binarization.
 */
export interface PostprocessOptions {
  /** Binarization threshold for pixel segmentation map. Default: 0.3. */
  threshold?: number;
  /** Polygon box threshold score for candidate filtering. Default: 0.6. */
  boxThreshold?: number;
  /** Maximum candidate boxes to detect per image. Default: 1000. */
  maxCandidates?: number;
  /** Polygon unclip expansion ratio. Default: 1.5 - 2.0. */
  unclipRatio?: number;
  /** Enable morphological dilation to connect broken text strokes. */
  useDilation?: boolean;
}

/**
 * Request-local OCR runtime execution options.
 */
export interface OcrRunOptions {
  /** Output formatting: 'lines' (default), 'words', or 'spatial' 2D layout. */
  output?: 'lines' | 'words' | 'spatial';
  /** Vertical center tolerance multiplier for same-line grouping. Default: 0.5. */
  lineYThreshold?: number;
  /** Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4. */
  wordXThreshold?: number;
  /** Minimum confidence score threshold [0.0 - 1.0] for recognition filtering. */
  textScore?: number;
  /** Request-local minimum height resize override. */
  minHeight?: number;
  /** Request-local maximum side length resize override. */
  maxSideLen?: number;
  /** Request-local minimum side length resize override. */
  minSideLen?: number;
  /** Width-to-height aspect ratio padding override. */
  widthHeightRatio?: number;
  /** Detection stage overrides. */
  detection?: DetectionOptions;
  /** Post-processing overrides. */
  postprocess?: PostprocessOptions;
}

/**
 * Configuration options for initializing the RustO engine session.
 */
export interface InitializeConfig {
  /** Model template preset: 'ppv6', 'ppv5', 'ppv4', or 'ppv3'. */
  preset?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';
  /** Model asset filesystem paths. */
  models?: {
    /** Detection model path (.rten or .onnx). */
    detection?: string;
    /** Recognition model path (.rten or .onnx). */
    recognition?: string;
    /** Character dictionary vocabulary file path (.txt). */
    dictionary?: string;
    /** Optional text-line orientation classifier model path. */
    classification?: string;
    /** Optional document orientation angle classifier model path. */
    orientation?: string;
  };
}

/**
 * High-performance, pure-Rust OCR engine for Node.js via native N-API addons.
 */
export class RustO {
  /**
   * Initializes the OCR engine with configuration options or a JSON configuration string.
   *
   * @param config - Initialization options or JSON string specifying model paths and presets.
   * @returns An initialized RustO engine instance ready for OCR operations.
   */
  static initialize(config: InitializeConfig | string): RustO;

  /**
   * Detects text in an image file located at the specified filesystem path.
   *
   * @param path - Filesystem path to the image file (JPEG, PNG, WebP, etc.).
   * @param options - Optional runtime detection options or JSON string.
   * @returns Array of structured TextResult objects with text, score, and bounding boxes.
   */
  detectTextFile(path: string, options?: OcrRunOptions | string): TextResult[];

  /**
   * Detects text from an in-memory image buffer.
   *
   * @param data - Raw image file data as Buffer or Uint8Array.
   * @param options - Optional runtime detection options or JSON string.
   * @returns Array of structured TextResult objects with text, score, and bounding boxes.
   */
  detectTextData(data: Buffer | Uint8Array, options?: OcrRunOptions | string): TextResult[];

  /**
   * Detects text in an image file and returns a reconstructed 2D spatial text layout.
   *
   * @param path - Filesystem path to the image file.
   * @param options - Optional runtime detection options or JSON string.
   * @returns Multi-line string preserving 2D spatial layout and spacing of original document.
   */
  detectTextSpatial(path: string, options?: OcrRunOptions | string): string;

  /**
   * Returns the underlying RustO engine version string.
   */
  version(): string;
}
