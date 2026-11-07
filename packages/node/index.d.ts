/**
 * Universal image source specification for Node.js:
 * - File path string
 * - Buffer or Uint8Array
 * - `{ path: string }`
 * - `{ uri: string }`
 * - `{ data: Buffer | Uint8Array }`
 * - `{ bytes: Buffer | Uint8Array }`
 */
export type ImageSource =
  | string
  | Buffer
  | Uint8Array
  | { path: string }
  | { uri: string }
  | { data: Buffer | Uint8Array }
  | { bytes: Buffer | Uint8Array };

/**
 * Semantic layout category of a detected document region.
 */
export type LayoutType =
  | 'text'
  | 'title'
  | 'figure'
  | 'figureCaption'
  | 'table'
  | 'tableCaption'
  | 'header'
  | 'footer'
  | 'reference'
  | 'equation';

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
  /** Optional detected document layout region index when layoutReading is enabled. */
  region_id?: number | null;
  /** Optional detected semantic layout category when layoutReading is enabled. */
  layout_type?: LayoutType | string | null;
}

/**
 * Output format granularity:
 * - 'lines': Group tokens into coherent text lines (default).
 * - 'words': Return individual word-level tokens.
 * - 'spatial': Return a formatted 2D text layout preserving spatial arrangement.
 */
export type OutputGranularity = 'lines' | 'words' | 'spatial' | 'tsv';

/**
 * Pre-detection image conditioning and resolution optimization options.
 */
export interface PreDetectionOptions {
  /** Descreen filtering strength (0.0 - 1.0). */
  descreenStrength?: number;
  /** Optimal target maximum side dimension for downscaling (e.g. 960 or 1024). */
  targetMaxSide?: number;
  /** Horizontal crop padding expansion ratio applied to bounding boxes (e.g. 0.05 = 5%). */
  cropPaddingX?: number;
  /** Vertical crop padding expansion ratio applied to bounding boxes (e.g. 0.05 = 5%). */
  cropPaddingY?: number;
}

/**
 * Image detection stage options.
 */
export interface DetectionRunOptions {
  /** Target side length constraint for image resizing. Default: 960. */
  limitSideLen?: number;
  /** Side length limit behavior: 'min' ensures short side meets threshold, 'max' caps long side. */
  limitType?: 'min' | 'max';
  /** Mean RGB normalization values. */
  mean?: [number, number, number];
  /** Standard deviation RGB normalization values. */
  std?: [number, number, number];
}

export type DetectionOptions = DetectionRunOptions;

/**
 * Post-detection polygon extraction, thresholding, and box filtering options.
 */
export interface PostDetectionOptions {
  /** Binarization threshold for pixel segmentation map. Default: 0.3. */
  threshold?: number;
  /** Polygon box threshold score for candidate filtering. Default: 0.6. */
  boxThreshold?: number;
  /** Polygon unclip expansion ratio. Default: 1.5 - 2.0. */
  unclipRatio?: number;
  /** Enable morphological dilation to connect broken text strokes. */
  useDilation?: boolean;
  /** DB score calculation mode: "fast" (bounding box mean) or "slow" (polygon contour mean). */
  scoreMode?: 'fast' | 'slow';
  /** Maximum candidate boxes to detect per image. Default: 1000. */
  maxCandidates?: number;
  /** Minimum text box height in pixels. */
  minHeight?: number;
  /** Minimum text box side length in pixels. */
  minSideLen?: number;
  /** Maximum text box side length in pixels. */
  maxSideLen?: number;
  /** Width-to-height aspect ratio filter cutoff. */
  widthHeightRatio?: number;
}

export type PostprocessOptions = PostDetectionOptions;

/**
 * Pre-recognition text line/word clustering tolerances.
 */
export interface PreRecognitionOptions {
  /** Vertical center tolerance multiplier for same-line grouping. Default: 0.5. */
  lineYThreshold?: number;
  /** Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4. */
  wordXThreshold?: number;
}

/**
 * Text recognition inference parameters.
 */
export interface RecognitionRunOptions {
  /** Recognition batch size (number of text crops processed concurrently). */
  batchSize?: number;
}

/**
 * Post-recognition confidence score filtering.
 */
export interface PostRecognitionOptions {
  /** Minimum confidence score threshold [0.0 - 1.0] for recognition filtering. */
  scoreThreshold?: number;
}

/**
 * Request-local OCR runtime execution options.
 */
export interface OcrRunOptions {
  /** Output formatting: 'lines' (default), 'words', or 'spatial' 2D layout. */
  output?: OutputGranularity;
  /** Enable document unwarping stage before detection. */
  autoUnwarp?: boolean;
  /** Enable document layout reading order analysis. */
  layoutReading?: boolean;
  /** Document layout or reading direction model identifier. */
  layoutModel?: string;
  /** Enable 0°/90°/180°/270° document orientation detection and rotation. */
  orientationDetection?: boolean;
  /** Enable 180° text line orientation classification. */
  lineClassification?: boolean;
  /** Phase 1: Pre-detection options (descreen, downscale, padding). */
  preDetection?: PreDetectionOptions;
  /** Phase 2: Detection inference options (limits, normalization). */
  detection?: DetectionRunOptions;
  /** Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry). */
  postDetection?: PostDetectionOptions;
  /** Phase 4: Pre-recognition grouping options (line_y, word_x tolerances). */
  preRecognition?: PreRecognitionOptions;
  /** Phase 5: Recognition inference options (batch size). */
  recognition?: RecognitionRunOptions;
  /** Phase 6: Post-recognition filtering options (score threshold). */
  postRecognition?: PostRecognitionOptions;
}

/**
 * Fluent builder for chaining text detection options and running inference.
 */
export class DetectTextBuilder<TResult = TextResult[]> {
  /** Specify the image source. */
  image(source: ImageSource): this;
  /** Specify a raw interleaved 24-bit RGB pixel buffer. */
  rgb(pixels: Buffer | Uint8Array, width: number, height: number, stride?: number): this;
  /** Enable or disable document unwarping. */
  withAutoUnwarp(enabled?: boolean): this;
  /** Enable or disable document layout reading order analysis. */
  withLayoutReading(enabled?: boolean): this;
  /** Specify document layout or reading direction model identifier. */
  withLayoutModel(model: string): this;
  /** Enable or disable document orientation detection. */
  withOrientationDetection(enabled?: boolean): this;
  /** Enable or disable text line orientation classification. */
  withLineClassification(enabled?: boolean): this;
  /** Configure pre-detection options. */
  withPreDetection(options: PreDetectionOptions): this;
  /** Configure detection inference options. */
  withDetection(options: DetectionRunOptions): this;
  /** Configure post-detection filtering options. */
  withPostDetection(options: PostDetectionOptions): this;
  /** Configure pre-recognition clustering options. */
  withPreRecognition(options: PreRecognitionOptions): this;
  /** Configure recognition inference options. */
  withRecognition(options: RecognitionRunOptions): this;
  /** Configure post-recognition filtering options. */
  withPostRecognition(options: PostRecognitionOptions): this;
  /** Set output format to line-level results. */
  asLines(): DetectTextBuilder<TextResult[]>;
  /** Set output format to word-level results. */
  asWords(): DetectTextBuilder<TextResult[]>;
  /** Set output format to 2D spatial text layout. */
  asSpatial(): DetectTextBuilder<string>;
  /** Set output format to TSV (tab-separated values). */
  asTsv(): DetectTextBuilder<string>;
  /** Set output granularity. */
  withOutput(granularity: 'spatial' | 'tsv'): DetectTextBuilder<string>;
  withOutput(granularity: 'lines' | 'words'): DetectTextBuilder<TextResult[]>;
  withOutput(granularity: OutputGranularity): DetectTextBuilder<TextResult[] | string>;
  /** Merge multiple options. */
  withOptions(options: OcrRunOptions & { output: 'spatial' | 'tsv' }): DetectTextBuilder<string>;
  withOptions(options: OcrRunOptions): this;
  /** Execute OCR detection with chained options. */
  run(source?: ImageSource): TResult;
}

/**
 * Configuration options for initializing the RustO engine session.
 */
export interface InitializeConfig {
  /** Model template preset: 'indra', 'ppv6', 'ppv5', 'ppv4', or 'ppv3'. */
  preset?: 'indra' | 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';
  /** Alias for preset. */
  template?: 'indra' | 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';
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
    /** Character dictionary vocabulary file path (.txt or .json). */
    dictionary?: string;
    /** Optional text-line orientation classifier model path. */
    classification?: string;
    /** Optional document orientation angle classifier model path. */
    orientation?: string;
    /** Optional document unwarping model path. */
    unwarp?: string;
    /** Optional document layout analysis model path. */
    layout?: string;
    /** Recognition batch size. Default: 6. */
    recBatchNum?: number;
  };
  /** Explicit detection model configuration. */
  detection?: {
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
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
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
    dictPath?: string;
    dict_path?: string;
    scoreThresh?: number;
    recBatchNum?: number;
  };
  /** Explicit classification model configuration. */
  classification?: {
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
    thresh?: number;
  };
  /** Explicit orientation model configuration. */
  orientation?: {
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
    thresh?: number;
  };
  /** Explicit unwarp model configuration. */
  unwarp?: {
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
  };
  /** Explicit document layout model configuration. */
  layout?: {
    enabled?: boolean;
    modelPath?: string;
    model_path?: string;
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
   * Constructs and initializes a new RustO OCR engine instance.
   *
   * @param config - Optional initialization options or JSON string. Defaults to Indra OCR (`indraocr`).
   */
  constructor(config?: InitializeConfig | string);

  /**
   * Initializes the OCR engine with configuration options or a JSON configuration string.
   *
   * @param config - Optional initialization options or JSON string specifying model paths and presets.
   * @returns An initialized RustO engine instance ready for OCR operations.
   */
  static initialize(config?: InitializeConfig | string): RustO;

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
   * Returns the underlying RustO engine version string.
   */
  static version(): string;

  /**
   * Checks whether an engine instance is ready for inference.
   */
  static isReady(engine?: RustO): boolean;

  /**
   * Starts a fluent builder for OCR detection.
   *
   * @returns Fluent builder chain ending with `.run()`.
   */
  detectText(): DetectTextBuilder;

  /**
   * Starts a fluent builder for OCR detection on the given image source.
   *
   * @param source - Image source (file path string, Buffer/Uint8Array, or source object).
   * @returns Fluent builder chain ending with `.run()`.
   */
  detectText(source: ImageSource): DetectTextBuilder;

  /**
   * Detects text in an image source with spatial text layout.
   */
  detectText(source: ImageSource, options: OcrRunOptions & { output: 'spatial' }): string;

  /**
   * Detects text in an image source with structured candidates.
   */
  detectText(source: ImageSource, options: OcrRunOptions & { output?: 'lines' | 'words' }): TextResult[];

  /**
   * Detects text in an image source.
   */
  detectText(source: ImageSource, options: OcrRunOptions | string): TextResult[] | string;

  /**
   * Detects text from a raw interleaved 24-bit RGB pixel buffer with spatial text layout.
   */
  detectTextRgb(
    pixels: Buffer | Uint8Array,
    width: number,
    height: number,
    stride: number | null | undefined,
    options: OcrRunOptions & { output: 'spatial' }
  ): string;

  /**
   * Detects text from a raw interleaved 24-bit RGB pixel buffer with structured candidates.
   */
  detectTextRgb(
    pixels: Buffer | Uint8Array,
    width: number,
    height: number,
    stride?: number | null,
    options?: (OcrRunOptions & { output?: 'lines' | 'words' }) | null
  ): TextResult[];

  /**
   * Detects text from a raw interleaved 24-bit RGB pixel buffer.
   */
  detectTextRgb(
    pixels: Buffer | Uint8Array,
    width: number,
    height: number,
    stride?: number | null,
    options?: OcrRunOptions | string | null
  ): TextResult[] | string;

  /**
   * Releases the underlying native engine handle.
   */
  close(): void;

  /**
   * Checks whether this engine instance is ready for inference.
   */
  isReady(): boolean;

  /**
   * Returns the underlying RustO engine version string.
   */
  version(): string;
}

export function version(): string;
export function isReady(engine?: RustO): boolean;

