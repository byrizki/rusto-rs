import type { DetectTextOptions, ModelPreset, OutputGranularity } from 'rusto-web';

export interface OcrModelOption {
  id: string;
  name: string;
  preset: ModelPreset;
  size: string;
  trait: string;
}

export const availableModels: OcrModelOption[] = [
  {
    id: 'ppocrv6-tiny',
    name: 'PP-OCRv6 Tiny',
    preset: 'ppv6',
    size: '~6 MB',
    trait: 'Ultra-lightweight',
  },
  {
    id: 'ppocrv6-small',
    name: 'PP-OCRv6 Small',
    preset: 'ppv6',
    size: '~30 MB',
    trait: 'Balanced speed & quality',
  },
  {
    id: 'ppocrv6-medium',
    name: 'PP-OCRv6 Medium',
    preset: 'ppv6',
    size: '~138 MB',
    trait: 'Highest accuracy',
  },
  {
    id: 'ppocrv5-mobile',
    name: 'PP-OCRv5 Mobile',
    preset: 'ppv5',
    size: '~12 MB',
    trait: 'Multi-lingual',
  },
  {
    id: 'ppocrv5-server',
    name: 'PP-OCRv5 Server',
    preset: 'ppv5',
    size: '~166 MB',
    trait: 'Server high accuracy',
  },
  {
    id: 'ppocrv4-mobile',
    name: 'PP-OCRv4 Mobile',
    preset: 'ppv4',
    size: '~16 MB',
    trait: 'Fast & compact',
  },
  {
    id: 'ppocrv4-server',
    name: 'PP-OCRv4 Server',
    preset: 'ppv4',
    size: '~196 MB',
    trait: 'Server high accuracy',
  },
];

export interface OcrConfig {
  // Confidence & filtering
  textScore: number;
  // Layout & line grouping
  lineYThreshold: number;
  wordXThreshold: number;
  // Stages
  classification: boolean;
  orientation: boolean;
  // Resizing overrides
  enableResize: boolean;
  minHeight: number;
  maxSideLen: number;
  minSideLen: number;
  widthHeightRatio: number;
  // Detection overrides
  detectionLimitSideLen: number;
  detectionLimitType: 'max' | 'min';
  // Postprocess overrides
  postprocessThreshold: number;
  postprocessBoxThreshold: number;
  postprocessUnclipRatio: number;
  postprocessMaxCandidates: number;
  postprocessUseDilation: boolean;
  // Calibration overrides
  calibrationEnabled: boolean;
  descreenBackground: boolean;
  descreenStrength: number;
  // Optimization overrides
  optimizationEnabled: boolean;
  optimizationTargetMaxSide: number;
  cropPaddingX: number;
  cropPaddingY: number;
}

export const defaultOcrConfig: OcrConfig = {
  textScore: 0.0,
  lineYThreshold: 0.5,
  wordXThreshold: 0.4,
  classification: false,
  orientation: false,
  enableResize: false,
  minHeight: 32,
  maxSideLen: 1920,
  minSideLen: 480,
  widthHeightRatio: -1,
  detectionLimitSideLen: 960,
  detectionLimitType: 'max',
  postprocessThreshold: 0.3,
  postprocessBoxThreshold: 0.5,
  postprocessUnclipRatio: 1.6,
  postprocessMaxCandidates: 1000,
  postprocessUseDilation: false,
  calibrationEnabled: true,
  descreenBackground: true,
  descreenStrength: 1.15,
  optimizationEnabled: true,
  optimizationTargetMaxSide: 1280,
  cropPaddingX: 10,
  cropPaddingY: 10,
};

export function buildDetectTextOptions(
  config: OcrConfig,
  outputMode: OutputGranularity
): DetectTextOptions {
  const isSpatial = outputMode === 'spatial';
  const opts: DetectTextOptions = {
    output: (isSpatial ? 'lines' : outputMode) as OutputGranularity,
    textScore: config.textScore,
    lineYThreshold: config.lineYThreshold,
    wordXThreshold: config.wordXThreshold,
    classification: config.classification,
    orientation: config.orientation,
    detection: {
      limitSideLen: config.detectionLimitSideLen,
      limitType: config.detectionLimitType,
    },
    postprocess: {
      threshold: config.postprocessThreshold,
      boxThreshold: config.postprocessBoxThreshold,
      unclipRatio: config.postprocessUnclipRatio,
      maxCandidates: config.postprocessMaxCandidates,
      useDilation: config.postprocessUseDilation,
    },
    calibration: {
      enabled: config.calibrationEnabled,
      descreenBackground: config.descreenBackground,
      descreenStrength: config.descreenStrength,
    },
    optimization: {
      enabled: config.optimizationEnabled,
      targetMaxSide: config.optimizationTargetMaxSide,
      cropPaddingX: config.cropPaddingX,
      cropPaddingY: config.cropPaddingY,
    },
  };

  if (config.enableResize) {
    if (config.minHeight > 0) opts.minHeight = config.minHeight;
    if (config.maxSideLen > 0) opts.maxSideLen = config.maxSideLen;
    if (config.minSideLen > 0 && config.minSideLen <= (config.maxSideLen || Infinity)) {
      opts.minSideLen = config.minSideLen;
    }
    if (config.widthHeightRatio > 0 || config.widthHeightRatio === -1) {
      opts.widthHeightRatio = config.widthHeightRatio;
    }
  }

  return opts;
}
