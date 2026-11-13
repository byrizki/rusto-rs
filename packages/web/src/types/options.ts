/**
 * Output format granularity:
 * - 'lines': Group tokens into coherent horizontal text lines (default).
 * - 'words': Return individual word-level tokens and bounding boxes.
 * - 'spatial': Return a formatted 2D text layout preserving spatial arrangement.
 */
export type OutputGranularity = 'lines' | 'words' | 'spatial' | 'tsv';

/**
 * Pretrained model architecture preset:
 * - 'ppv6': PP-OCRv6 multilingual 50+ languages model (recommended).
 * - 'ppv5': PP-OCRv5 model family.
 * - 'ppv4': PP-OCRv4 model family.
 * - 'ppv3': PP-OCRv3 legacy model family.
 * - 'indra': Indra OCR Indonesian ID & Document model family (DataLab).
 */
export type ModelPreset = 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3' | 'indra';

/**
 * Phase 1: Pre-detection options (descreen, downscale, padding).
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
 * Phase 2: Detection inference options (limits, normalization).
 */
export interface DetectionRunOptions {
  /** Target side length constraint for image resizing. Default: 960. */
  limitSideLen?: number;
  /** Side length limit behavior: 'min' ensures short side meets threshold, 'max' caps long side. */
  limitType?: 'min' | 'max';
  /** Mean RGB normalization values. Default: [0.485, 0.456, 0.406]. */
  mean?: [number, number, number];
  /** Standard deviation RGB normalization values. Default: [0.229, 0.224, 0.225]. */
  std?: [number, number, number];
}

export type DetectionOptions = DetectionRunOptions;

/**
 * Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry).
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
 * Pre-OCR image calibration options for descreening patterned backgrounds.
 */
export interface CalibrationOptions {
  /** Enable background descreening calibration. */
  enabled?: boolean;
  /** Descreen periodic halftone dot patterns and security patterns. */
  descreenBackground?: boolean;
  /** Descreen filtering strength (0.0 - 1.0). */
  descreenStrength?: number;
}

/**
 * Runtime crop and resolution optimization options.
 */
export interface OptimizationOptions {
  /** Enable runtime ROI cropping and scaling optimization. */
  enabled?: boolean;
  /** Target maximum side length for cropped candidate patches. */
  targetMaxSide?: number;
  /** Horizontal padding around bounding box crops in pixels. */
  cropPaddingX?: number;
  /** Vertical padding around bounding box crops in pixels. */
  cropPaddingY?: number;
}

/**
 * Phase 4: Pre-recognition grouping options (line_y, word_x tolerances).
 */
export interface PreRecognitionOptions {
  /** Vertical center tolerance multiplier for same-line grouping. Default: 0.5. */
  lineYThreshold?: number;
  /** Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4. */
  wordXThreshold?: number;
}

/**
 * Phase 5: Recognition inference options (batch size and worker concurrency).
 */
export interface RecognitionRunOptions {
  /** Recognition batch size (number of text crops processed concurrently). */
  batchSize?: number;
  /** Number of parallel worker threads for recognition inference (undefined = auto). */
  workers?: number;
  /** Maximum worker thread count clamp for recognition inference (undefined = engine default 8). */
  maxWorkers?: number;
}

/**
 * Phase 6: Post-recognition filtering options (score threshold).
 */
export interface PostRecognitionOptions {
  /** Minimum confidence score threshold [0.0 - 1.0] for recognition filtering. */
  scoreThreshold?: number;
}

/**
 * Per-request execution options for text detection.
 */
export interface OcrRunOptions {
  /** Output granularity: 'lines' (default), 'words', or 'spatial'. */
  output?: OutputGranularity;
  /** Enable document unwarping stage before detection. */
  autoUnwarp?: boolean;
  /** Enable document layout reading order analysis. */
  layoutReading?: boolean;
  /** Layout / reading order architecture variant or model identifier (e.g. 'indra', 'cdla', 'publaynet', 'heuristic', 'none'). */
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

export type DetectTextOptions = OcrRunOptions;
