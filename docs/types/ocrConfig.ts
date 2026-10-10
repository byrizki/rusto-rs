import type { DetectTextOptions, ModelPreset, OutputGranularity } from '@rusto/web';

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
    id: 'ppocrv6-tiny-int8',
    name: 'PP-OCRv6 Tiny (INT8)',
    preset: 'ppv6',
    size: '~3.4 MB',
    trait: 'Ultra-lightweight INT8',
  },
  {
    id: 'ppocrv6-small',
    name: 'PP-OCRv6 Small',
    preset: 'ppv6',
    size: '~30 MB',
    trait: 'Balanced speed & quality',
  },
  {
    id: 'ppocrv6-small-int8',
    name: 'PP-OCRv6 Small (INT8)',
    preset: 'ppv6',
    size: '~16 MB',
    trait: 'Balanced INT8',
  },
  {
    id: 'ppocrv6-medium',
    name: 'PP-OCRv6 Medium',
    preset: 'ppv6',
    size: '~138 MB',
    trait: 'Highest accuracy',
  },
  {
    id: 'ppocrv6-medium-int8',
    name: 'PP-OCRv6 Medium (INT8)',
    preset: 'ppv6',
    size: '~76 MB',
    trait: 'Highest accuracy INT8',
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

export const defaultModel: OcrModelOption = availableModels[0] as OcrModelOption;

export interface SampleImage {
  id: string;
  name: string;
  description: string;
  url: string;
}

export function getSampleImages(baseURL: string): SampleImage[] {
  return [
    {
      id: 'invoice',
      name: 'Invoice',
      description: 'Structured layout with numbers & tables',
      url: `${baseURL}samples/example1.png`,
    },
    {
      id: 'idcard',
      name: 'Passport',
      description: 'Identity document with personal details & MRZ',
      url: `${baseURL}samples/idcard.jpg`,
    },
    {
      id: 'handwritten',
      name: 'Handwritten',
      description: 'Vintage cursive handwriting receipt',
      url: `${baseURL}samples/invoice1.jpg`,
    },
  ];
}

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
