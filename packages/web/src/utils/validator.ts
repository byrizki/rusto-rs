import type {
  DetectTextOptions,
  InitializeConfig,
  ModelPreset,
} from '../types/index.js';

const OPTION_KEYS = [
  'output',
  'lineYThreshold',
  'wordXThreshold',
  'textScore',
  'classification',
  'orientation',
  'minHeight',
  'maxSideLen',
  'minSideLen',
  'widthHeightRatio',
  'detection',
  'postprocess',
  'calibration',
  'optimization',
] as const;

const MODEL_KEYS = [
  'detection',
  'recognition',
  'dictionary',
  'classification',
  'orientation',
] as const;

const DETECTION_KEYS = ['limitSideLen', 'limitType', 'mean', 'std'] as const;

const POSTPROCESS_KEYS = [
  'threshold',
  'boxThreshold',
  'maxCandidates',
  'unclipRatio',
  'useDilation',
] as const;

const CALIBRATION_KEYS = [
  'enabled',
  'descreenBackground',
  'descreenStrength',
] as const;

const OPTIMIZATION_KEYS = [
  'enabled',
  'targetMaxSide',
  'cropPaddingX',
  'cropPaddingY',
] as const;

const PRESETS: readonly ModelPreset[] = ['ppv6', 'ppv5', 'ppv4', 'ppv3'] as const;

export function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && value !== undefined && typeof value === 'object' && !Array.isArray(value);
}

export function requireOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  message: string
): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) {
    throw new TypeError(message);
  }
}

export function normalizeDetectionOptions(detection: unknown): void {
  if (detection === undefined) return;
  if (!isObject(detection)) {
    throw new TypeError('DetectTextOptions.detection must be an object.');
  }
  requireOnlyKeys(
    detection,
    DETECTION_KEYS,
    'DetectTextOptions.detection contains an unknown key.'
  );

  const limitSideLen = detection.limitSideLen;
  if (
    limitSideLen !== undefined &&
    (typeof limitSideLen !== 'number' ||
      !Number.isInteger(limitSideLen) ||
      limitSideLen < 1 ||
      limitSideLen > 32767)
  ) {
    throw new TypeError(
      'DetectTextOptions.detection.limitSideLen must be an integer between 1 and 32767.'
    );
  }

  if (
    detection.limitType !== undefined &&
    detection.limitType !== 'min' &&
    detection.limitType !== 'max'
  ) {
    throw new TypeError('DetectTextOptions.detection.limitType is invalid.');
  }

  for (const key of ['mean', 'std'] as const) {
    const values = detection[key];
    if (
      values !== undefined &&
      (!Array.isArray(values) ||
        values.length !== 3 ||
        values.some(
          (value) =>
            typeof value !== 'number' || !Number.isFinite(value) || (key === 'std' && value === 0)
        ))
    ) {
      throw new TypeError(`DetectTextOptions.detection.${key} must contain three valid numbers.`);
    }
  }
}

export function normalizePostprocessOptions(postprocess: unknown): void {
  if (postprocess === undefined) return;
  if (!isObject(postprocess)) {
    throw new TypeError('DetectTextOptions.postprocess must be an object.');
  }
  requireOnlyKeys(
    postprocess,
    POSTPROCESS_KEYS,
    'DetectTextOptions.postprocess contains an unknown key.'
  );

  for (const key of ['threshold', 'boxThreshold'] as const) {
    const value = postprocess[key];
    if (
      value !== undefined &&
      (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)
    ) {
      throw new TypeError(`DetectTextOptions.postprocess.${key} must be in [0, 1].`);
    }
  }

  const maxCandidates = postprocess.maxCandidates;
  if (
    maxCandidates !== undefined &&
    (typeof maxCandidates !== 'number' || !Number.isInteger(maxCandidates) || maxCandidates < 1)
  ) {
    throw new TypeError('DetectTextOptions.postprocess.maxCandidates must be an integer >= 1.');
  }

  if (
    postprocess.unclipRatio !== undefined &&
    (typeof postprocess.unclipRatio !== 'number' ||
      !Number.isFinite(postprocess.unclipRatio) ||
      postprocess.unclipRatio <= 0)
  ) {
    throw new TypeError('DetectTextOptions.postprocess.unclipRatio must be > 0.');
  }

  if (postprocess.useDilation !== undefined && typeof postprocess.useDilation !== 'boolean') {
    throw new TypeError('DetectTextOptions.postprocess.useDilation must be a boolean.');
  }
}

export function normalizeOptions(options: DetectTextOptions | undefined): DetectTextOptions {
  if (options === undefined) return {};
  if (!isObject(options)) throw new TypeError('DetectTextOptions must be an object.');
  requireOnlyKeys(options, OPTION_KEYS, 'DetectTextOptions contains an unknown key.');

  const output = options.output;
  if (output !== undefined && !['lines', 'words', 'spatial'].includes(output as string)) {
    throw new TypeError('DetectTextOptions.output is invalid.');
  }

  for (const key of ['lineYThreshold', 'wordXThreshold'] as const) {
    const value = options[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0)) {
      throw new TypeError(`DetectTextOptions.${key} must be a finite number >= 0.`);
    }
  }

  const score = options.textScore;
  if (
    score !== undefined &&
    (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)
  ) {
    throw new TypeError('DetectTextOptions.textScore must be a finite number from 0 to 1.');
  }

  for (const key of ['classification', 'orientation'] as const) {
    const value = options[key];
    if (value !== undefined && typeof value !== 'boolean') {
      throw new TypeError(`DetectTextOptions.${key} must be a boolean.`);
    }
  }

  for (const key of ['minHeight', 'maxSideLen', 'minSideLen'] as const) {
    const value = options[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)) {
      throw new TypeError(`DetectTextOptions.${key} must be a finite number > 0.`);
    }
  }

  const ratio = options.widthHeightRatio;
  if (
    ratio !== undefined &&
    (typeof ratio !== 'number' || !Number.isFinite(ratio) || (ratio <= 0 && ratio !== -1))
  ) {
    throw new TypeError('DetectTextOptions.widthHeightRatio must be > 0 or -1.');
  }

  const minSideLen = options.minSideLen;
  const maxSideLen = options.maxSideLen;
  if (typeof minSideLen === 'number' && typeof maxSideLen === 'number' && minSideLen > maxSideLen) {
    throw new TypeError('DetectTextOptions.minSideLen must be <= maxSideLen.');
  }

  normalizeDetectionOptions(options.detection);
  normalizePostprocessOptions(options.postprocess);
  normalizeCalibrationOptions(options.calibration);
  normalizeOptimizationOptions(options.optimization);
  return options;
}

export function normalizeCalibrationOptions(calibration: unknown): void {
  if (calibration === undefined) return;
  if (!isObject(calibration)) {
    throw new TypeError('DetectTextOptions.calibration must be an object.');
  }
  requireOnlyKeys(
    calibration,
    CALIBRATION_KEYS,
    'DetectTextOptions.calibration contains an unknown key.'
  );
  if (calibration.enabled !== undefined && typeof calibration.enabled !== 'boolean') {
    throw new TypeError('DetectTextOptions.calibration.enabled must be a boolean.');
  }
  if (
    calibration.descreenBackground !== undefined &&
    typeof calibration.descreenBackground !== 'boolean'
  ) {
    throw new TypeError('DetectTextOptions.calibration.descreenBackground must be a boolean.');
  }
  if (
    calibration.descreenStrength !== undefined &&
    (typeof calibration.descreenStrength !== 'number' ||
      !Number.isFinite(calibration.descreenStrength) ||
      calibration.descreenStrength <= 0)
  ) {
    throw new TypeError('DetectTextOptions.calibration.descreenStrength must be a positive number.');
  }
}

export function normalizeOptimizationOptions(optimization: unknown): void {
  if (optimization === undefined) return;
  if (!isObject(optimization)) {
    throw new TypeError('DetectTextOptions.optimization must be an object.');
  }
  requireOnlyKeys(
    optimization,
    OPTIMIZATION_KEYS,
    'DetectTextOptions.optimization contains an unknown key.'
  );
  if (optimization.enabled !== undefined && typeof optimization.enabled !== 'boolean') {
    throw new TypeError('DetectTextOptions.optimization.enabled must be a boolean.');
  }
}

export function normalizeInitializeConfig(config: InitializeConfig | undefined): InitializeConfig {
  if (config === undefined) return {};
  if (!isObject(config)) throw new TypeError('InitializeConfig must be an object.');

  if (config.preset !== undefined && !PRESETS.includes(config.preset as ModelPreset)) {
    throw new TypeError('InitializeConfig.preset is invalid.');
  }

  if (config.models !== undefined) {
    if (!isObject(config.models)) throw new TypeError('InitializeConfig.models must be an object.');
    requireOnlyKeys(config.models, MODEL_KEYS, 'InitializeConfig.models contains an unknown key.');
    for (const [key, value] of Object.entries(config.models)) {
      if (value === undefined) continue;
      const isString = typeof value === 'string' && value.trim().length > 0;
      const isBuffer = value instanceof Uint8Array || value instanceof ArrayBuffer;
      if (!isString && !isBuffer) {
        throw new TypeError(`InitializeConfig.models.${key} must be a non-empty string or Uint8Array/ArrayBuffer.`);
      }
    }
  }

  return config;
}

export function validateImageSource(source: unknown): void {
  if (!source) {
    throw new TypeError('ImageSource must be provided and not empty.');
  }
}
