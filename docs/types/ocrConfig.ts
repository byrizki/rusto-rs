import type { DetectTextOptions, ModelPreset, OutputGranularity } from '@rustors/web';

export interface ProgressStep {
  current: number;
  total: number;
}

export interface ProgressState {
  active: boolean;
  phase: 'download' | 'init' | 'inference';
  stageText: string;
  percent: number;
  detailText: string;
  step?: ProgressStep;
}

export const defaultProgressState: ProgressState = {
  active: false,
  phase: 'download',
  stageText: '',
  percent: 0,
  detailText: '',
};

export interface OcrConfig {
  // Confidence & filtering
  textScore: number;
  // Layout & reading order
  lineYThreshold: number;
  wordXThreshold: number;
  enableReadingOrder: boolean;
  // Recognition batching
  recBatchNum: number;
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
  textScore: 0.5,
  lineYThreshold: 0.5,
  wordXThreshold: 0.4,
  enableReadingOrder: false,
  recBatchNum: 1,
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
  postprocessUnclipRatio: 1.8,
  postprocessMaxCandidates: 1000,
  postprocessUseDilation: false,
  calibrationEnabled: true,
  descreenBackground: true,
  descreenStrength: 1.15,
  optimizationEnabled: true,
  optimizationTargetMaxSide: 960,
  cropPaddingX: 0,
  cropPaddingY: 0.02,
};

export interface ActiveRunConfig {
  modelId: string;
  modelName: string;
  modelPreset: ModelPreset;
  outputMode: 'lines' | 'words';
  timestamp: number;
  durationMs: number | null;
  itemsCount: number;
  options: DetectTextOptions;
  rawConfig: OcrConfig;
}

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
    enableReadingOrder: config.enableReadingOrder,
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
