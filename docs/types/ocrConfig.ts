import type { DetectTextOptions, ModelPreset, OutputGranularity } from '@rustors/web';
import type { LayoutModelId } from './ocrModels';

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
  // Phase 6: Post-recognition filtering
  textScore: number;
  // Phase 4: Pre-recognition grouping
  lineYThreshold: number;
  wordXThreshold: number;
  // Auxiliary: Layout & reading order
  layoutReading: boolean;
  enableReadingOrder: boolean;
  layoutModel: LayoutModelId;
  // Phase 5: Recognition batching
  recBatchNum: number;
  // Auxiliary: Document & line transforms
  autoUnwarp: boolean;
  lineClassification: boolean;
  classification?: boolean;
  orientationDetection: boolean;
  orientation?: boolean;
  // Phase 3: Post-detection overrides & resizing
  enableResize: boolean;
  minHeight: number;
  maxSideLen: number;
  minSideLen: number;
  widthHeightRatio: number;
  // Phase 2: Detection overrides
  detectionLimitSideLen: number;
  detectionLimitType: 'max' | 'min';
  // Phase 3: Post-detection DBNet thresholds
  postprocessThreshold: number;
  postprocessBoxThreshold: number;
  postprocessUnclipRatio: number;
  postprocessMaxCandidates: number;
  postprocessUseDilation: boolean;
  // Phase 1: Pre-detection overrides (calibration & descreening)
  calibrationEnabled: boolean;
  descreenBackground: boolean;
  descreenStrength: number;
  // Phase 1: Pre-detection optimization & downscale
  optimizationEnabled: boolean;
  optimizationTargetMaxSide: number;
  cropPaddingX: number;
  cropPaddingY: number;
  // Inference Engine Backend
  inferenceBackend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';
  useWebGpu: boolean;
}

export const defaultOcrConfig: OcrConfig = {
  inferenceBackend: 'rusto-nn',
  textScore: 0.5,
  lineYThreshold: 0.5,
  wordXThreshold: 0.4,
  layoutReading: false,
  enableReadingOrder: false,
  layoutModel: 'heuristic',
  recBatchNum: 1,
  autoUnwarp: false,
  lineClassification: false,
  classification: false,
  orientationDetection: false,
  orientation: false,
  enableResize: false,
  minHeight: 32,
  maxSideLen: 1920,
  minSideLen: 480,
  widthHeightRatio: -1,
  detectionLimitSideLen: 960,
  detectionLimitType: 'min',
  postprocessThreshold: 0.3,
  postprocessBoxThreshold: 0.6,
  postprocessUnclipRatio: 2.0,
  postprocessMaxCandidates: 1000,
  postprocessUseDilation: true,
  calibrationEnabled: false,
  descreenBackground: true,
  descreenStrength: 1.15,
  optimizationEnabled: false,
  optimizationTargetMaxSide: 800,
  cropPaddingX: 0,
  cropPaddingY: 0.02,
  useWebGpu: false,
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

function buildPreDetectionConfig(config: OcrConfig): Record<string, unknown> | undefined {
  const pre: Record<string, unknown> = {};
  if (config.calibrationEnabled && config.descreenBackground && config.descreenStrength > 0) {
    pre.descreenStrength = config.descreenStrength;
  }
  if (config.optimizationEnabled) {
    if (config.optimizationTargetMaxSide > 0) {
      pre.targetMaxSide = config.optimizationTargetMaxSide;
    }
    if (config.cropPaddingX > 0) pre.cropPaddingX = config.cropPaddingX;
    if (config.cropPaddingY > 0) pre.cropPaddingY = config.cropPaddingY;
  }
  return Object.keys(pre).length > 0 ? pre : undefined;
}

function buildPostDetectionConfig(config: OcrConfig): Record<string, unknown> {
  const post: Record<string, unknown> = {
    threshold: config.postprocessThreshold,
    boxThreshold: config.postprocessBoxThreshold,
    unclipRatio: config.postprocessUnclipRatio,
    maxCandidates: config.postprocessMaxCandidates,
    useDilation: config.postprocessUseDilation,
  };
  if (config.enableResize) {
    if (config.minHeight > 0) post.minHeight = config.minHeight;
    if (config.maxSideLen > 0) post.maxSideLen = config.maxSideLen;
    if (config.minSideLen > 0 && config.minSideLen <= (config.maxSideLen || Infinity)) {
      post.minSideLen = config.minSideLen;
    }
    if (config.widthHeightRatio > 0 || config.widthHeightRatio === -1) {
      post.widthHeightRatio = config.widthHeightRatio;
    }
  }
  return post;
}

export function isReadingOrderEnabled(config: OcrConfig): boolean {
  return (
    config.layoutModel !== 'none' &&
    Boolean(config.layoutReading || config.enableReadingOrder || config.layoutModel !== undefined)
  );
}

export function buildDetectTextOptions(
  config: OcrConfig,
  outputMode: OutputGranularity
): DetectTextOptions {
  const isSpatial = outputMode === 'spatial';
  const isReadingOrderActive = isReadingOrderEnabled(config);

  return {
    output: (isSpatial ? 'lines' : outputMode) as OutputGranularity,
    autoUnwarp: Boolean(config.autoUnwarp),
    layoutReading: isReadingOrderActive,
    layoutModel: config.layoutModel,
    orientationDetection: Boolean(config.orientationDetection ?? config.orientation),
    lineClassification: Boolean(config.lineClassification ?? config.classification),
    preDetection: buildPreDetectionConfig(config),
    detection: {
      limitSideLen: config.detectionLimitSideLen,
      limitType: config.detectionLimitType,
    },
    postDetection: buildPostDetectionConfig(config),
    preRecognition:
      config.layoutModel === 'heuristic'
        ? {
            lineYThreshold: config.lineYThreshold,
            wordXThreshold: config.wordXThreshold,
          }
        : undefined,
    recognition: {
      batchSize: config.recBatchNum,
    },
    postRecognition: {
      scoreThreshold: config.textScore,
    },
  };
}
