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
}
export interface Frame { width: number; height: number; top: number; left: number; }
export interface TextResult { text: string; score: number; box_points: [[number, number], [number, number], [number, number], [number, number]]; frame: Frame; }

const SOURCE_KEYS = ['uri', 'base64', 'bytes'] as const;
const OPTION_KEYS = ['output', 'lineYThreshold', 'wordXThreshold', 'textScore', 'classification', 'orientation'] as const;
const MODEL_KEYS = ['detection', 'recognition', 'dictionary', 'classification', 'orientation'] as const;
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
  requireOnlyKeys(source, SOURCE_KEYS, 'ImageSource must contain exactly one of uri, base64, or bytes.');
  const keys = Object.keys(source);
  if (keys.length !== 1 || !SOURCE_KEYS.includes(keys[0] as typeof SOURCE_KEYS[number])) {
    throw new TypeError('ImageSource must contain exactly one of uri, base64, or bytes.');
  }
  const key = keys[0] as typeof SOURCE_KEYS[number];
  const value = source[key];
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
