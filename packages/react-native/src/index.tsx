import { NativeModules, Platform } from 'react-native';

const LINKING_ERROR =
  `The package 'react-native-rusto' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

const Rusto = NativeModules.Rusto
  ? NativeModules.Rusto
  : new Proxy({}, { get() { throw new Error(LINKING_ERROR); } });

export type BinaryImageData = Uint8Array | ArrayBuffer;
export type ImageSource = { uri: string } | { base64: string } | { bytes: BinaryImageData };
export type OutputGranularity = 'lines' | 'words' | 'spatial';
export type ModelPreset = 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';

export interface PostprocessOptions {
  threshold?: number;
  boxThreshold?: number;
  maxCandidates?: number;
  unclipRatio?: number;
  useDilation?: boolean;
}
export interface DetectionPreprocessingOptions {
  limitSideLen?: number;
  limitType?: 'min' | 'max';
  mean?: [number, number, number];
  std?: [number, number, number];
  postprocess?: PostprocessOptions;
}
export interface PreprocessingOptions {
  minHeight?: number;
  maxSideLen?: number;
  minSideLen?: number;
  widthHeightRatio?: number;
  detection?: DetectionPreprocessingOptions;
}
export interface InitializeConfig {
  preset?: ModelPreset;
  models?: { detection?: string; recognition?: string; dictionary?: string; classification?: string; orientation?: string };
}
export interface DetectTextOptions {
  output?: OutputGranularity;
  lineYThreshold?: number;
  wordXThreshold?: number;
  textScore?: number;
  classification?: boolean;
  orientation?: boolean;
  /** Request-local preprocessing and detector postprocess overrides. */
  preprocessing?: PreprocessingOptions;
}
export interface Frame { width: number; height: number; top: number; left: number; }
export interface TextResult { text: string; score: number; box_points: [[number, number], [number, number], [number, number], [number, number]]; frame: Frame; }

const SOURCE_KEYS = ['uri', 'base64', 'bytes'] as const;
const OPTION_KEYS = ['output', 'lineYThreshold', 'wordXThreshold', 'textScore', 'classification', 'orientation', 'preprocessing'] as const;
const MODEL_KEYS = ['detection', 'recognition', 'dictionary', 'classification', 'orientation'] as const;
const PREPROCESSING_KEYS = ['minHeight', 'maxSideLen', 'minSideLen', 'widthHeightRatio', 'detection'] as const;
const DETECTION_PREPROCESSING_KEYS = ['limitSideLen', 'limitType', 'mean', 'std', 'postprocess'] as const;
const POSTPROCESS_KEYS = ['threshold', 'boxThreshold', 'maxCandidates', 'unclipRatio', 'useDilation'] as const;
const PRESETS = ['ppv6', 'ppv5', 'ppv4', 'ppv3'] as const;

function isObject(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}
function requireOnlyKeys(value: Record<string, unknown>, allowed: readonly string[], message: string): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) throw new TypeError(message);
}
function bytesToBase64(bytes: unknown): string {
  if (!(bytes instanceof Uint8Array) && !(bytes instanceof ArrayBuffer)) {
    throw new TypeError('ImageSource.bytes must be a Uint8Array or ArrayBuffer.');
  }
  const data = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
  if (data.byteLength === 0) throw new TypeError('ImageSource.bytes must not be empty.');
  let binary = '';
  for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
  const encoder = global.btoa;
  if (typeof encoder !== 'function') throw new Error('Base64 encoding is unavailable in this React Native runtime.');
  return encoder(binary);
}
function normalizeSource(source: ImageSource): { uri?: string; base64?: string } {
  if (!isObject(source)) throw new TypeError('ImageSource must be an object with exactly one of uri, base64, or bytes.');
  // Runtime validation deliberately handles values cast past ImageSource's union type.
  const rawSource = source as unknown as Record<string, unknown>;
  requireOnlyKeys(rawSource, SOURCE_KEYS, 'ImageSource must contain exactly one of uri, base64, or bytes.');
  const keys = Object.keys(rawSource);
  if (keys.length !== 1 || !SOURCE_KEYS.includes(keys[0] as typeof SOURCE_KEYS[number])) {
    throw new TypeError('ImageSource must contain exactly one of uri, base64, or bytes.');
  }
  const key = keys[0] as typeof SOURCE_KEYS[number];
  const value = rawSource[key];
  if (key === 'bytes') return { base64: bytesToBase64(value) };
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`ImageSource.${key} must be a non-empty string.`);
  }
  return { [key]: value.trim() };
}
function normalizeOptions(options: DetectTextOptions | undefined): DetectTextOptions {
  if (options === undefined) return {};
  if (!isObject(options)) throw new TypeError('DetectTextOptions must be an object.');
  requireOnlyKeys(options, OPTION_KEYS, 'DetectTextOptions contains an unknown key.');
  const output = options.output;
  if (output !== undefined && (!['lines', 'words', 'spatial'].includes(output as string))) throw new TypeError('DetectTextOptions.output is invalid.');
  for (const key of ['lineYThreshold', 'wordXThreshold'] as const) {
    const value = options[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0)) throw new TypeError(`DetectTextOptions.${key} must be a finite number >= 0.`);
  }
  const score = options.textScore;
  if (score !== undefined && (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)) throw new TypeError('DetectTextOptions.textScore must be a finite number from 0 to 1.');
  for (const key of ['classification', 'orientation'] as const) {
    const value = options[key];
    if (value !== undefined && typeof value !== 'boolean') throw new TypeError(`DetectTextOptions.${key} must be a boolean.`);
  }
  const preprocessing = options.preprocessing;
  if (preprocessing !== undefined) {
    if (!isObject(preprocessing)) throw new TypeError('DetectTextOptions.preprocessing must be an object.');
    requireOnlyKeys(preprocessing, PREPROCESSING_KEYS, 'DetectTextOptions.preprocessing contains an unknown key.');
    for (const key of ['minHeight', 'maxSideLen', 'minSideLen'] as const) {
      const value = preprocessing[key];
      if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)) throw new TypeError(`DetectTextOptions.preprocessing.${key} must be a finite number > 0.`);
    }
    const ratio = preprocessing.widthHeightRatio;
    if (ratio !== undefined && (typeof ratio !== 'number' || !Number.isFinite(ratio) || (ratio <= 0 && ratio !== -1))) throw new TypeError('DetectTextOptions.preprocessing.widthHeightRatio must be > 0 or -1.');
    if (preprocessing.minSideLen !== undefined && preprocessing.maxSideLen !== undefined && preprocessing.minSideLen > preprocessing.maxSideLen) throw new TypeError('DetectTextOptions.preprocessing.minSideLen must be <= maxSideLen.');
    const detection = preprocessing.detection;
    if (detection !== undefined) {
      if (!isObject(detection)) throw new TypeError('DetectTextOptions.preprocessing.detection must be an object.');
      requireOnlyKeys(detection, DETECTION_PREPROCESSING_KEYS, 'DetectTextOptions.preprocessing.detection contains an unknown key.');
      if (detection.limitSideLen !== undefined && (!Number.isInteger(detection.limitSideLen) || detection.limitSideLen < 1 || detection.limitSideLen > 32767)) throw new TypeError('DetectTextOptions.preprocessing.detection.limitSideLen must be an integer between 1 and 32767.');
      if (detection.limitType !== undefined && detection.limitType !== 'min' && detection.limitType !== 'max') throw new TypeError('DetectTextOptions.preprocessing.detection.limitType is invalid.');
      for (const key of ['mean', 'std'] as const) {
        const values = detection[key];
        if (values !== undefined && (!Array.isArray(values) || values.length !== 3 || values.some((value) => typeof value !== 'number' || !Number.isFinite(value) || (key === 'std' && value === 0)))) throw new TypeError(`DetectTextOptions.preprocessing.detection.${key} must contain three valid numbers.`);
      }
      const postprocess = detection.postprocess;
      if (postprocess !== undefined) {
        if (!isObject(postprocess)) throw new TypeError('DetectTextOptions.preprocessing.detection.postprocess must be an object.');
        requireOnlyKeys(postprocess, POSTPROCESS_KEYS, 'DetectTextOptions.preprocessing.detection.postprocess contains an unknown key.');
        for (const key of ['threshold', 'boxThreshold'] as const) { const value = postprocess[key]; if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) throw new TypeError(`DetectTextOptions.preprocessing.detection.postprocess.${key} must be in [0, 1].`); }
        if (postprocess.maxCandidates !== undefined && (!Number.isInteger(postprocess.maxCandidates) || postprocess.maxCandidates < 1)) throw new TypeError('DetectTextOptions.preprocessing.detection.postprocess.maxCandidates must be an integer >= 1.');
        if (postprocess.unclipRatio !== undefined && (typeof postprocess.unclipRatio !== 'number' || !Number.isFinite(postprocess.unclipRatio) || postprocess.unclipRatio <= 0)) throw new TypeError('DetectTextOptions.preprocessing.detection.postprocess.unclipRatio must be > 0.');
        if (postprocess.useDilation !== undefined && typeof postprocess.useDilation !== 'boolean') throw new TypeError('DetectTextOptions.preprocessing.detection.postprocess.useDilation must be a boolean.');
      }
    }
  }
  return options;
}
function normalizeInitializeConfig(config: InitializeConfig | undefined): InitializeConfig {
  if (config === undefined) return {};
  if (!isObject(config)) throw new TypeError('InitializeConfig must be an object.');
  requireOnlyKeys(config, ['preset', 'models'], 'InitializeConfig contains an unknown key.');
  if (config.preset !== undefined && !PRESETS.includes(config.preset as ModelPreset)) throw new TypeError('InitializeConfig.preset is invalid.');
  if (config.models !== undefined) {
    if (!isObject(config.models)) throw new TypeError('InitializeConfig.models must be an object.');
    requireOnlyKeys(config.models, MODEL_KEYS, 'InitializeConfig.models contains an unknown key.');
    for (const [key, value] of Object.entries(config.models)) {
      if (typeof value !== 'string' || value.trim().length === 0) throw new TypeError(`InitializeConfig.models.${key} must be a non-empty string.`);
    }
  }
  return config;
}

export function initialize(config: InitializeConfig = {}): Promise<void> { return Rusto.initialize(normalizeInitializeConfig(config)); }
export function detectText(source: ImageSource, options: DetectTextOptions & { output: 'spatial' }): Promise<string>;
export function detectText(source: ImageSource, options?: DetectTextOptions & { output?: 'lines' | 'words' }): Promise<TextResult[]>;
export function detectText(source: ImageSource, options: DetectTextOptions = {}): Promise<TextResult[] | string> {
  return Rusto.detectText(normalizeSource(source), normalizeOptions(options));
}
