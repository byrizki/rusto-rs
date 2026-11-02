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
 * Pre-OCR image calibration options for descreening patterned backgrounds.
 */
export interface CalibrationOptions {
  /** Descreen filtering strength (0.0 - 1.0). */
  descreenStrength?: number;
}

/**
 * Runtime crop and resolution optimization options.
 */
export interface OptimizationOptions {
  /** Enable or disable image optimization. */
  enabled?: boolean;
  /** Optimal target maximum side dimension for downscaling (e.g. 960 or 1024). */
  targetMaxSide?: number;
  /** Horizontal crop padding expansion ratio applied to bounding boxes (e.g. 0.06 = 6%). */
  cropPaddingX?: number;
  /** Vertical crop padding expansion ratio applied to bounding boxes (e.g. 0.08 = 8%). */
  cropPaddingY?: number;
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
  /** Recognition batch size override (number of text lines processed per inference batch). */
  recBatchNum?: number;
  /** Pre-OCR background descreening calibration options. */
  calibration?: CalibrationOptions;
  /** Runtime crop and resolution optimization options. */
  optimization?: OptimizationOptions;
}

/**
 * Configuration options for initializing the RustO engine session.
 */
export interface InitializeConfig {
  /** Model template preset: 'ppv6', 'ppv5', 'ppv4', or 'ppv3'. */
  preset?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';
  /** Canonical model identifier or name (e.g. 'indraocr' (default), 'ppocrv6-tiny', 'ppocrv6-small', 'ppocrv5', 'ppocrv4'). */
  model?: string;
  /** Directory path containing det, rec, and dict model files. */
  modelPath?: string;
  /** Alias for modelPath. */
  modelDir?: string;
  /** Language identifier (e.g. 'latin', 'cyrillic', 'ar', 'ja', 'ko', 'devanagari'). */
  lang?: string;
  /** Custom cache directory override path. */
  cacheDir?: string;
  /** Enable neural reading order recovery transformer. */
  enableReadingOrder?: boolean;
  /** Document layout variant: 'cdla' (default 10-class) or 'publaynet' (5-class). */
  layoutVariant?: string;
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
    /** Recognition batch size. Default: 6. */
    recBatchNum?: number;
  };
  /** Explicit detection model configuration. */
  detection?: {
    modelPath?: string;
    limitSideLen?: number;
    limitType?: 'min' | 'max';
    thresh?: number;
    boxThresh?: number;
    unclipRatio?: number;
    useDilation?: boolean;
    scoreMode?: 'fast' | 'slow';
  };
  /** Explicit recognition model configuration. */
  recognition?: {
    modelPath?: string;
    dictPath?: string;
    scoreThresh?: number;
    recBatchNum?: number;
  };
  /** Explicit classification model configuration. */
  classification?: {
    enabled?: boolean;
    modelPath?: string;
    thresh?: number;
  };
  /** Explicit orientation model configuration. */
  orientation?: {
    enabled?: boolean;
    modelPath?: string;
    thresh?: number;
  };
  /** Explicit unwarp model configuration. */
  unwarp?: {
    enabled?: boolean;
    modelPath?: string;
  };
  /** Explicit document layout model configuration. */
  layout?: {
    enabled?: boolean;
    modelPath?: string;
    confThresh?: number;
    targetSize?: number;
    targetHeight?: number;
    targetWidth?: number;
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
   * Downloads model assets for a specific OCR pipeline ahead of time.
   *
   * @param model - Canonical model name (e.g. 'indraocr', 'ppocrv6-tiny', 'ppocrv5'). Defaults to 'indraocr'.
   * @param lang - Optional language identifier (e.g. 'latin', 'cyrillic', 'ar', 'ja').
   * @param cacheDir - Optional custom cache directory.
   * @returns Array of downloaded local file paths.
   */
  static downloadModel(model?: string, lang?: string, cacheDir?: string): string[];

  /**
   * Downloads all available canonical models to the cache directory ahead of time.
   *
   * @param cacheDir - Optional custom cache directory.
   * @returns Array of downloaded local file paths.
   */
  static downloadAllModels(cacheDir?: string): string[];

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
