/**
 * Output format granularity:
 * - 'lines': Group tokens into coherent horizontal text lines (default).
 * - 'words': Return individual word-level tokens and bounding boxes.
 * - 'spatial': Return a formatted 2D text layout preserving spatial arrangement.
 */
export type OutputGranularity = 'lines' | 'words' | 'spatial';

/**
 * Pretrained model architecture preset:
 * - 'ppv6': PP-OCRv6 multilingual 50+ languages model (recommended).
 * - 'ppv5': PP-OCRv5 model family.
 * - 'ppv4': PP-OCRv4 model family.
 * - 'ppv3': PP-OCRv3 legacy model family.
 */
export type ModelPreset = 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';

/**
 * Image detection stage options.
 */
export interface DetectionOptions {
  /** Target side length constraint for image resizing. Default: 960. */
  limitSideLen?: number;
  /** Side length limit behavior: 'min' ensures short side meets threshold, 'max' caps long side. */
  limitType?: 'min' | 'max';
  /** Mean RGB normalization values. Default: [0.485, 0.456, 0.406]. */
  mean?: [number, number, number];
  /** Standard deviation RGB normalization values. Default: [0.229, 0.224, 0.225]. */
  std?: [number, number, number];
}

/**
 * Post-processing options for DBNet polygon detection and binarization.
 */
export interface PostprocessOptions {
  /** Binarization probability cutoff threshold for segmentation map. Default: 0.3. */
  threshold?: number;
  /** Box score confidence threshold for candidate contour filtering. Default: 0.6. */
  boxThreshold?: number;
  /** Maximum candidate text boxes to detect per image. Default: 1000. */
  maxCandidates?: number;
  /** Polygon unclip expansion ratio. Default: 1.5 - 2.0. */
  unclipRatio?: number;
  /** Enable morphological dilation to connect broken or faint text strokes. */
  useDilation?: boolean;
}

/**
 * Pre-OCR image calibration options for descreening patterned backgrounds (e.g. ID cards, banknotes).
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
 * Per-request execution options for text detection.
 */
export interface DetectTextOptions {
  /** Output granularity: 'lines' (default), 'words', or 'spatial'. */
  output?: OutputGranularity;

  /** Same-row vertical center tolerance multiplier. Default: 0.5. */
  lineYThreshold?: number;

  /** Same-word horizontal gap tolerance multiplier. Default: 0.4. */
  wordXThreshold?: number;

  /** Reject recognized candidates below this confidence score [0, 1]. */
  textScore?: number;

  /** Enable classification stage. Default: false. */
  classification?: boolean;

  /** Enable orientation correction stage. Default: false. */
  orientation?: boolean;

  /** Request-local minimum height resize override. */
  minHeight?: number;

  /** Request-local maximum side length resize override. */
  maxSideLen?: number;

  /** Request-local minimum side length resize override. */
  minSideLen?: number;

  /** Request-local width-to-height ratio override. */
  widthHeightRatio?: number;

  /** Request-local detector input overrides. */
  detection?: DetectionOptions;

  /** Request-local detector postprocess overrides. */
  postprocess?: PostprocessOptions;

  /** Pre-OCR image calibration (descreening patterned backgrounds, e.g. ID cards). Default: true. */
  calibration?: CalibrationOptions;

  /** Runtime crop and resolution optimization. */
  optimization?: OptimizationOptions;
}
