import { NativeModules, Platform } from 'react-native';

const LINKING_ERROR =
  `The package '@rustors/react-native' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

const Rusto = NativeModules.Rusto
  ? NativeModules.Rusto
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    );

/**
 * Binary image payload represented as Uint8Array or ArrayBuffer.
 */
export type BinaryImageData = Uint8Array | ArrayBuffer;

/**
 * Universal image source specification for React Native:
 * - `{ uri: string }`: Local `file://` URI or app asset path.
 * - `{ base64: string }`: Base64-encoded image string.
 * - `{ bytes: BinaryImageData }`: Raw binary byte array or buffer.
 */
export type ImageSource = { uri: string } | { base64: string } | { bytes: BinaryImageData };

/**
 * Output format granularity:
 * - 'lines': Group tokens into coherent text lines (default).
 * - 'words': Return individual word-level tokens.
 * - 'spatial': Return a formatted 2D text layout preserving spatial arrangement.
 */
export type OutputGranularity = 'lines' | 'words' | 'spatial';

/**
 * Model architecture preset: 'ppv6', 'ppv5', 'ppv4', or 'ppv3'.
 */
export type ModelPreset = 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';

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
  /** Enable morphological dilation to connect broken text strokes. */
  useDilation?: boolean;
  /** DB score calculation mode: "fast" (bounding box mean) or "slow" (polygon contour mean). */
  scoreMode?: 'fast' | 'slow';
}

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
 * Configuration options for initializing the RustO engine session in React Native.
 */
export interface InitializeConfig {
  /** Model architecture template preset. Defaults to 'ppv6'. */
  preset?: ModelPreset;
  /** Custom model asset file paths (bundled in app or downloaded). */
  models?: {
    /** Detection model path (.rten or .onnx). */
    detection?: string;
    /** Recognition model path (.rten or .onnx). */
    recognition?: string;
    /** Character dictionary vocabulary file path (.txt). */
    dictionary?: string;
    /** Optional text-line orientation classifier model path. */
    classification?: string;
    /** Optional document orientation angle classifier model path. */
    orientation?: string;
  };
}

/**
 * Request-local OCR runtime execution options.
 */
export interface DetectTextOptions {
  /** Output granularity: 'lines' (default), 'words', or 'spatial'. */
  output?: OutputGranularity;
  /** Same-row vertical center tolerance multiplier. Default: 0.5. */
  lineYThreshold?: number;
  /** Same-word horizontal gap tolerance multiplier. Default: 0.4. */
  wordXThreshold?: number;
  /** Minimum recognition confidence score threshold [0.0 - 1.0]. */
  textScore?: number;
  /** Enable 180° text line orientation classifier stage. */
  classification?: boolean;
  /** Enable 0°/90°/180°/270° document orientation rotation stage. */
  orientation?: boolean;
  /** Request-local minimum height resize override. */
  minHeight?: number;
  /** Request-local maximum side length resize override. */
  maxSideLen?: number;
  /** Request-local minimum side length resize override. */
  minSideLen?: number;
  /** Width-to-height aspect ratio padding override. */
  widthHeightRatio?: number;
  /** Request-local detector input overrides. */
  detection?: DetectionOptions;
  /** Request-local detector postprocess overrides. */
  postprocess?: PostprocessOptions;
}

/**
 * Axis-aligned bounding box frame with position and size in image pixels.
 */
export interface Frame {
  /** Width of the bounding box in pixels. */
  width: number;
  /** Height of the bounding box in pixels. */
  height: number;
  /** Top vertical coordinate (Y) of the bounding box. */
  top: number;
  /** Left horizontal coordinate (X) of the bounding box. */
  left: number;
}

/**
 * Recognized OCR text result containing text, confidence score, bounding polygon, and frame.
 */
export interface TextResult {
  /** Recognized text string. */
  text: string;
  /** Confidence score between 0.0 and 1.0. */
  score: number;
  /** Four polygon vertex coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]. */
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  /** Axis-aligned bounding box frame. */
  frame: Frame;
}

const SOURCE_KEYS = ['uri', 'base64', 'bytes'] as const;
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
  'scoreMode',
] as const;
const PRESETS = ['ppv6', 'ppv5', 'ppv4', 'ppv3'] as const;

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && value !== undefined && typeof value === 'object' && !Array.isArray(value);
}
function requireOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  message: string
): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) throw new TypeError(message);
}
function bytesToBase64(bytes: unknown): string {
  if (!(bytes instanceof Uint8Array) && !(bytes instanceof ArrayBuffer)) {
    throw new TypeError('ImageSource.bytes must be a Uint8Array or ArrayBuffer.');
  }
  const data = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
  if (data.byteLength === 0) throw new TypeError('ImageSource.bytes must not be empty.');
  let binary = '';
  for (let i = 0; i < data.length; i += 0x8000)
    binary += String.fromCodePoint(...data.subarray(i, i + 0x8000));
  const encoder = global.btoa;
  if (typeof encoder !== 'function')
    throw new Error('Base64 encoding is unavailable in this React Native runtime.');
  return encoder(binary);
}
function normalizeSource(source: ImageSource): { uri?: string; base64?: string } {
  if (!isObject(source))
    throw new TypeError('ImageSource must be an object with exactly one of uri, base64, or bytes.');
  // Runtime validation deliberately handles values cast past ImageSource's union type.
  const rawSource = source as unknown as Record<string, unknown>;
  requireOnlyKeys(
    rawSource,
    SOURCE_KEYS,
    'ImageSource must contain exactly one of uri, base64, or bytes.'
  );
  const keys = Object.keys(rawSource);
  if (keys.length !== 1 || !SOURCE_KEYS.includes(keys[0] as (typeof SOURCE_KEYS)[number])) {
    throw new TypeError('ImageSource must contain exactly one of uri, base64, or bytes.');
  }
  const key = keys[0] as (typeof SOURCE_KEYS)[number];
  const value = rawSource[key];
  if (key === 'bytes') return { base64: bytesToBase64(value) };
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`ImageSource.${key} must be a non-empty string.`);
  }
  return { [key]: value.trim() };
}
function normalizeDetectionOptions(detection: unknown): void {
  if (detection === undefined) return;
  if (!isObject(detection)) throw new TypeError('DetectTextOptions.detection must be an object.');
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
  )
    throw new TypeError(
      'DetectTextOptions.detection.limitSideLen must be an integer between 1 and 32767.'
    );
  if (
    detection.limitType !== undefined &&
    detection.limitType !== 'min' &&
    detection.limitType !== 'max'
  )
    throw new TypeError('DetectTextOptions.detection.limitType is invalid.');
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
    )
      throw new TypeError(`DetectTextOptions.detection.${key} must contain three valid numbers.`);
  }
}
function normalizePostprocessOptions(postprocess: unknown): void {
  if (postprocess === undefined) return;
  if (!isObject(postprocess))
    throw new TypeError('DetectTextOptions.postprocess must be an object.');
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
    )
      throw new TypeError(`DetectTextOptions.postprocess.${key} must be in [0, 1].`);
  }
  const maxCandidates = postprocess.maxCandidates;
  if (
    maxCandidates !== undefined &&
    (typeof maxCandidates !== 'number' || !Number.isInteger(maxCandidates) || maxCandidates < 1)
  )
    throw new TypeError('DetectTextOptions.postprocess.maxCandidates must be an integer >= 1.');
  if (
    postprocess.unclipRatio !== undefined &&
    (typeof postprocess.unclipRatio !== 'number' ||
      !Number.isFinite(postprocess.unclipRatio) ||
      postprocess.unclipRatio <= 0)
  )
    throw new TypeError('DetectTextOptions.postprocess.unclipRatio must be > 0.');
  if (postprocess.useDilation !== undefined && typeof postprocess.useDilation !== 'boolean')
    throw new TypeError('DetectTextOptions.postprocess.useDilation must be a boolean.');
  if (
    postprocess.scoreMode !== undefined &&
    postprocess.scoreMode !== 'fast' &&
    postprocess.scoreMode !== 'slow'
  )
    throw new TypeError('DetectTextOptions.postprocess.scoreMode must be "fast" or "slow".');
}
function normalizeOptions(options: DetectTextOptions | undefined): DetectTextOptions {
  if (options === undefined) return {};
  if (!isObject(options)) throw new TypeError('DetectTextOptions must be an object.');
  requireOnlyKeys(options, OPTION_KEYS, 'DetectTextOptions contains an unknown key.');
  const output = options.output;
  if (output !== undefined && !['lines', 'words', 'spatial'].includes(output as string))
    throw new TypeError('DetectTextOptions.output is invalid.');
  for (const key of ['lineYThreshold', 'wordXThreshold'] as const) {
    const value = options[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0))
      throw new TypeError(`DetectTextOptions.${key} must be a finite number >= 0.`);
  }
  const score = options.textScore;
  if (
    score !== undefined &&
    (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)
  )
    throw new TypeError('DetectTextOptions.textScore must be a finite number from 0 to 1.');
  for (const key of ['classification', 'orientation'] as const) {
    const value = options[key];
    if (value !== undefined && typeof value !== 'boolean')
      throw new TypeError(`DetectTextOptions.${key} must be a boolean.`);
  }
  for (const key of ['minHeight', 'maxSideLen', 'minSideLen'] as const) {
    const value = options[key];
    if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0))
      throw new TypeError(`DetectTextOptions.${key} must be a finite number > 0.`);
  }
  const ratio = options.widthHeightRatio;
  if (
    ratio !== undefined &&
    (typeof ratio !== 'number' || !Number.isFinite(ratio) || (ratio <= 0 && ratio !== -1))
  )
    throw new TypeError('DetectTextOptions.widthHeightRatio must be > 0 or -1.');
  const minSideLen = options.minSideLen;
  const maxSideLen = options.maxSideLen;
  if (typeof minSideLen === 'number' && typeof maxSideLen === 'number' && minSideLen > maxSideLen)
    throw new TypeError('DetectTextOptions.minSideLen must be <= maxSideLen.');
  normalizeDetectionOptions(options.detection);
  normalizePostprocessOptions(options.postprocess);
  return options;
}
function normalizeInitializeConfig(config: InitializeConfig | undefined): InitializeConfig {
  if (config === undefined) return {};
  if (!isObject(config)) throw new TypeError('InitializeConfig must be an object.');
  requireOnlyKeys(config, ['preset', 'models'], 'InitializeConfig contains an unknown key.');
  if (config.preset !== undefined && !PRESETS.includes(config.preset as ModelPreset))
    throw new TypeError('InitializeConfig.preset is invalid.');
  if (config.models !== undefined) {
    if (!isObject(config.models)) throw new TypeError('InitializeConfig.models must be an object.');
    requireOnlyKeys(config.models, MODEL_KEYS, 'InitializeConfig.models contains an unknown key.');
    for (const [key, value] of Object.entries(config.models)) {
      if (typeof value !== 'string' || value.trim().length === 0)
        throw new TypeError(`InitializeConfig.models.${key} must be a non-empty string.`);
    }
  }
  return config;
}

/**
 * Initializes the native RustO OCR engine in React Native.
 *
 * @param config - Optional configuration with preset ('ppv6', etc.) and custom model paths.
 * @returns Promise that resolves when the native model sessions are loaded.
 */
export function initialize(config: InitializeConfig = {}): Promise<void> {
  return Rusto.initialize(normalizeInitializeConfig(config));
}

/**
 * Runs OCR text recognition on an image source with reconstructed 2D spatial text output.
 *
 * @param source - Image source object ({ uri }, { base64 }, or { bytes }).
 * @param options - Execution options with `output: 'spatial'`.
 * @returns Formatted 2D multi-line text layout.
 */
export function detectText(
  source: ImageSource,
  options: DetectTextOptions & { output: 'spatial' }
): Promise<string>;

/**
 * Runs OCR text recognition on an image source with structured line or word candidates.
 *
 * @param source - Image source object ({ uri }, { base64 }, or { bytes }).
 * @param options - Execution options with `output: 'lines'` or `output: 'words'`.
 * @returns Array of structured TextResult objects.
 */
export function detectText(
  source: ImageSource,
  options?: DetectTextOptions & { output?: 'lines' | 'words' }
): Promise<TextResult[]>;

/**
 * Runs OCR text recognition on an image source.
 *
 * @param source - Image source object ({ uri }, { base64 }, or { bytes }).
 * @param options - Optional execution options.
 * @returns Structured TextResult array or spatial string depending on output option.
 */
export function detectText(
  source: ImageSource,
  options?: DetectTextOptions
): Promise<TextResult[] | string>;

export function detectText(
  source: ImageSource,
  options: DetectTextOptions = {}
): Promise<TextResult[] | string> {
  return Rusto.detectText(normalizeSource(source), normalizeOptions(options));
}
