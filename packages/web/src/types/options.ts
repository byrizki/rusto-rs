export type OutputGranularity = 'lines' | 'words' | 'spatial';

export type ModelPreset = 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';

export interface DetectionOptions {
  limitSideLen?: number;
  limitType?: 'min' | 'max';
  mean?: [number, number, number];
  std?: [number, number, number];
}

export interface PostprocessOptions {
  threshold?: number;
  boxThreshold?: number;
  maxCandidates?: number;
  unclipRatio?: number;
  useDilation?: boolean;
}

export interface CalibrationOptions {
  enabled?: boolean;
  descreenBackground?: boolean;
  descreenStrength?: number;
}

export interface OptimizationOptions {
  enabled?: boolean;
  targetMaxSide?: number;
  cropPaddingX?: number;
  cropPaddingY?: number;
}

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
