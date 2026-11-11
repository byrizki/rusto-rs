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
export type OutputGranularity = 'lines' | 'words' | 'spatial' | 'tsv';

/**
 * Model architecture preset: 'indra', 'ppv6', 'ppv5', 'ppv4', or 'ppv3'.
 */
export type ModelPreset = 'indra' | 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';

/**
 * Phase 1: Pre-detection options (descreen, downscale, padding).
 */
export interface PreDetectionOptions {
  /** Descreen filtering strength (0.0 - 1.0). */
  descreenStrength?: number;
  /** Optimal target maximum side dimension for downscaling (e.g. 960 or 1024). */
  targetMaxSide?: number;
  /** Horizontal crop padding expansion ratio applied to bounding boxes (e.g. 0.05 = 5%). */
  cropPaddingX?: number;
  /** Vertical crop padding expansion ratio applied to bounding boxes (e.g. 0.05 = 5%). */
  cropPaddingY?: number;
}

/**
 * Phase 2: Detection inference options (limits, normalization).
 */
export interface DetectionRunOptions {
  /** Target side length constraint for image resizing. Default: 960. */
  limitSideLen?: number;
  /** Side length limit behavior: 'min' ensures short side meets threshold, 'max' caps long side. */
  limitType?: 'min' | 'max';
  /** Mean RGB normalization values. Default: [0.485, 0.456, 0.406]. */
  mean?: [number, number, number];
  /** Standard deviation RGB normalization values. Default: [0.229, 0.224, 0.225]. */
  std?: [number, number, number];
}

export type DetectionOptions = DetectionRunOptions;

/**
 * Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry).
 */
export interface PostDetectionOptions {
  /** Binarization threshold for pixel segmentation map. Default: 0.3. */
  threshold?: number;
  /** Polygon box threshold score for candidate filtering. Default: 0.6. */
  boxThreshold?: number;
  /** Polygon unclip expansion ratio. Default: 1.5 - 2.0. */
  unclipRatio?: number;
  /** Enable morphological dilation to connect broken text strokes. */
  useDilation?: boolean;
  /** DB score calculation mode: "fast" (bounding box mean) or "slow" (polygon contour mean). */
  scoreMode?: 'fast' | 'slow';
  /** Maximum candidate boxes to detect per image. Default: 1000. */
  maxCandidates?: number;
  /** Minimum text box height in pixels. */
  minHeight?: number;
  /** Minimum text box side length in pixels. */
  minSideLen?: number;
  /** Maximum text box side length in pixels. */
  maxSideLen?: number;
  /** Width-to-height aspect ratio filter cutoff. */
  widthHeightRatio?: number;
}

export type PostprocessOptions = PostDetectionOptions;

/**
 * Phase 4: Pre-recognition grouping options (line_y, word_x tolerances).
 */
export interface PreRecognitionOptions {
  /** Vertical center tolerance multiplier for same-line grouping. Default: 0.5. */
  lineYThreshold?: number;
  /** Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4. */
  wordXThreshold?: number;
}

/**
 * Phase 5: Recognition inference options (batch size).
 */
export interface RecognitionRunOptions {
  /** Recognition batch size (number of text crops processed concurrently). */
  batchSize?: number;
}

/**
 * Phase 6: Post-recognition filtering options (score threshold).
 */
export interface PostRecognitionOptions {
  /** Minimum confidence score threshold [0.0 - 1.0] for recognition filtering. */
  scoreThreshold?: number;
}

/**
 * Request-local OCR runtime execution options.
 */
export interface OcrRunOptions {
  /** Output granularity: 'lines' (default), 'words', or 'spatial'. */
  output?: OutputGranularity;
  /** Enable document unwarping stage before detection. */
  autoUnwarp?: boolean;
  /** Enable document layout reading order analysis. */
  layoutReading?: boolean;
  /** Document layout or reading direction model identifier. */
  layoutModel?: string;
  /** Enable 0°/90°/180°/270° document orientation detection and rotation. */
  orientationDetection?: boolean;
  /** Enable 180° text line orientation classification. */
  lineClassification?: boolean;
  /** Phase 1: Pre-detection options (descreen, downscale, padding). */
  preDetection?: PreDetectionOptions;
  /** Phase 2: Detection inference options (limits, normalization). */
  detection?: DetectionRunOptions;
  /** Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry). */
  postDetection?: PostDetectionOptions;
  /** Phase 4: Pre-recognition grouping options (line_y, word_x tolerances). */
  preRecognition?: PreRecognitionOptions;
  /** Phase 5: Recognition inference options (batch size). */
  recognition?: RecognitionRunOptions;
  /** Phase 6: Post-recognition filtering options (score threshold). */
  postRecognition?: PostRecognitionOptions;
}

export type DetectTextOptions = OcrRunOptions;

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
  /** Optional layout region index if layoutReading was enabled. */
  region_id?: number;
  /** Optional layout semantic tag if layoutReading was enabled. */
  layout_type?: string;
}

/**
 * Configuration options for initializing the RustO engine session in React Native.
 */
export interface InitializeConfig {
  /** Model architecture template preset. Defaults to 'ppv6'. */
  preset?: ModelPreset;
  /** Model pipeline identifier: 'ppocrv6-tiny' (default), 'ppocrv6-tiny-fp16', 'ppocrv6-small', 'indraocr', 'ppocrv5', 'ppocrv4'. */
  model?: string;
  /** Base model directory containing det, rec, and dict files. */
  modelPath?: string;
  /** Target language: 'latin' (default), 'ja', 'korean', 'ar', 'cyrillic', 'devanagari', etc. */
  lang?: string;
  /** Custom model cache directory path. */
  cacheDir?: string;
  /** Enable neural reading order recovery transformer. */
  enableReadingOrder?: boolean;
  /** Document layout variant: 'cdla' (default 10-class) or 'publaynet' (5-class). */
  layoutVariant?: 'cdla' | 'publaynet';
  /** Optional recognition batch size override. Default: 6. */
  recBatchNum?: number;
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
    /** Optional document unwarping model path (.rten or .onnx). */
    unwarp?: string;
    /** Optional document layout analysis model path (.rten or .onnx). */
    layout?: string;
  };
}

const SOURCE_KEYS = ['uri', 'base64', 'bytes'] as const;
const OPTION_KEYS = [
  'output',
  'autoUnwarp',
  'layoutReading',
  'layoutModel',
  'orientationDetection',
  'lineClassification',
  'preDetection',
  'detection',
  'postDetection',
  'preRecognition',
  'recognition',
  'postRecognition',
] as const;
const MODEL_KEYS = [
  'detection',
  'recognition',
  'dictionary',
  'classification',
  'orientation',
  'unwarp',
  'layout',
] as const;
const PRE_DETECTION_KEYS = [
  'descreenStrength',
  'targetMaxSide',
  'cropPaddingX',
  'cropPaddingY',
] as const;
const DETECTION_KEYS = ['limitSideLen', 'limitType', 'mean', 'std'] as const;
const POST_DETECTION_KEYS = [
  'threshold',
  'boxThreshold',
  'unclipRatio',
  'useDilation',
  'scoreMode',
  'maxCandidates',
  'minHeight',
  'minSideLen',
  'maxSideLen',
  'widthHeightRatio',
] as const;
const PRE_RECOGNITION_KEYS = ['lineYThreshold', 'wordXThreshold'] as const;
const RECOGNITION_KEYS = ['batchSize'] as const;
const POST_RECOGNITION_KEYS = ['scoreThreshold'] as const;
const PRESETS = ['indra', 'ppv6', 'ppv5', 'ppv4', 'ppv3'] as const;

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
  const encoder = (global as any).btoa;
  if (typeof encoder !== 'function')
    throw new Error('Base64 encoding is unavailable in this React Native runtime.');
  return encoder(binary);
}

function normalizeSource(source: ImageSource): { uri?: string; base64?: string } {
  if (!isObject(source))
    throw new TypeError('ImageSource must be an object with exactly one of uri, base64, or bytes.');
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

function normalizePreDetectionOptions(preDetection: unknown): void {
  if (preDetection === undefined) return;
  if (!isObject(preDetection)) throw new TypeError('OcrRunOptions.preDetection must be an object.');
  requireOnlyKeys(
    preDetection,
    PRE_DETECTION_KEYS,
    'OcrRunOptions.preDetection contains an unknown key.'
  );
  const descreen = preDetection.descreenStrength;
  if (descreen !== undefined && (typeof descreen !== 'number' || !Number.isFinite(descreen) || descreen < 0))
    throw new TypeError('OcrRunOptions.preDetection.descreenStrength must be a non-negative number.');
  const target = preDetection.targetMaxSide;
  if (target !== undefined && (typeof target !== 'number' || !Number.isInteger(target) || target < 1))
    throw new TypeError('OcrRunOptions.preDetection.targetMaxSide must be an integer >= 1.');
  for (const key of ['cropPaddingX', 'cropPaddingY'] as const) {
    const pad = preDetection[key];
    if (pad !== undefined && (typeof pad !== 'number' || !Number.isFinite(pad) || pad < 0))
      throw new TypeError(`OcrRunOptions.preDetection.${key} must be a non-negative number.`);
  }
}

function normalizeDetectionOptions(detection: unknown): void {
  if (detection === undefined) return;
  if (!isObject(detection)) throw new TypeError('OcrRunOptions.detection must be an object.');
  requireOnlyKeys(
    detection,
    DETECTION_KEYS,
    'OcrRunOptions.detection contains an unknown key.'
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
      'OcrRunOptions.detection.limitSideLen must be an integer between 1 and 32767.'
    );
  if (
    detection.limitType !== undefined &&
    detection.limitType !== 'min' &&
    detection.limitType !== 'max'
  )
    throw new TypeError('OcrRunOptions.detection.limitType is invalid.');
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
      throw new TypeError(`OcrRunOptions.detection.${key} must contain three valid numbers.`);
  }
}

function normalizePostDetectionOptions(postDetection: unknown): void {
  if (postDetection === undefined) return;
  if (!isObject(postDetection))
    throw new TypeError('OcrRunOptions.postDetection must be an object.');
  requireOnlyKeys(
    postDetection,
    POST_DETECTION_KEYS,
    'OcrRunOptions.postDetection contains an unknown key.'
  );
  for (const key of ['threshold', 'boxThreshold'] as const) {
    const value = postDetection[key];
    if (
      value !== undefined &&
      (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)
    )
      throw new TypeError(`OcrRunOptions.postDetection.${key} must be in [0, 1].`);
  }
  const maxCandidates = postDetection.maxCandidates;
  if (
    maxCandidates !== undefined &&
    (typeof maxCandidates !== 'number' || !Number.isInteger(maxCandidates) || maxCandidates < 1)
  )
    throw new TypeError('OcrRunOptions.postDetection.maxCandidates must be an integer >= 1.');
  if (
    postDetection.unclipRatio !== undefined &&
    (typeof postDetection.unclipRatio !== 'number' ||
      !Number.isFinite(postDetection.unclipRatio) ||
      postDetection.unclipRatio <= 0)
  )
    throw new TypeError('OcrRunOptions.postDetection.unclipRatio must be > 0.');
  if (postDetection.useDilation !== undefined && typeof postDetection.useDilation !== 'boolean')
    throw new TypeError('OcrRunOptions.postDetection.useDilation must be a boolean.');
  if (
    postDetection.scoreMode !== undefined &&
    postDetection.scoreMode !== 'fast' &&
    postDetection.scoreMode !== 'slow'
  )
    throw new TypeError('OcrRunOptions.postDetection.scoreMode must be "fast" or "slow".');
  for (const key of ['minHeight', 'maxSideLen', 'minSideLen'] as const) {
    const val = postDetection[key];
    if (val !== undefined && (typeof val !== 'number' || !Number.isFinite(val) || val <= 0))
      throw new TypeError(`OcrRunOptions.postDetection.${key} must be a finite number > 0.`);
  }
  const ratio = postDetection.widthHeightRatio;
  if (ratio !== undefined && (typeof ratio !== 'number' || !Number.isFinite(ratio) || (ratio <= 0 && ratio !== -1)))
    throw new TypeError('OcrRunOptions.postDetection.widthHeightRatio must be > 0 or -1.');
  const minSideLen = postDetection.minSideLen;
  const maxSideLen = postDetection.maxSideLen;
  if (typeof minSideLen === 'number' && typeof maxSideLen === 'number' && minSideLen > maxSideLen)
    throw new TypeError('OcrRunOptions.postDetection.minSideLen must be <= maxSideLen.');
}

function normalizePreRecognitionOptions(preRecognition: unknown): void {
  if (preRecognition === undefined) return;
  if (!isObject(preRecognition)) throw new TypeError('OcrRunOptions.preRecognition must be an object.');
  requireOnlyKeys(
    preRecognition,
    PRE_RECOGNITION_KEYS,
    'OcrRunOptions.preRecognition contains an unknown key.'
  );
  for (const key of ['lineYThreshold', 'wordXThreshold'] as const) {
    const val = preRecognition[key];
    if (val !== undefined && (typeof val !== 'number' || !Number.isFinite(val) || val < 0))
      throw new TypeError(`OcrRunOptions.preRecognition.${key} must be a finite number >= 0.`);
  }
}

function normalizeRecognitionOptions(recognition: unknown): void {
  if (recognition === undefined) return;
  if (!isObject(recognition)) throw new TypeError('OcrRunOptions.recognition must be an object.');
  requireOnlyKeys(
    recognition,
    RECOGNITION_KEYS,
    'OcrRunOptions.recognition contains an unknown key.'
  );
  const batchSize = recognition.batchSize;
  if (batchSize !== undefined && (typeof batchSize !== 'number' || !Number.isInteger(batchSize) || batchSize < 1))
    throw new TypeError('OcrRunOptions.recognition.batchSize must be an integer >= 1.');
}

function normalizePostRecognitionOptions(postRecognition: unknown): void {
  if (postRecognition === undefined) return;
  if (!isObject(postRecognition)) throw new TypeError('OcrRunOptions.postRecognition must be an object.');
  requireOnlyKeys(
    postRecognition,
    POST_RECOGNITION_KEYS,
    'OcrRunOptions.postRecognition contains an unknown key.'
  );
  const score = postRecognition.scoreThreshold;
  if (score !== undefined && (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1))
    throw new TypeError('OcrRunOptions.postRecognition.scoreThreshold must be a finite number from 0 to 1.');
}

function normalizeOptions(options: OcrRunOptions | undefined): OcrRunOptions {
  if (options === undefined) return {};
  if (!isObject(options)) throw new TypeError('OcrRunOptions must be an object.');
  requireOnlyKeys(options, OPTION_KEYS, 'OcrRunOptions contains an unknown key.');
  const output = options.output;
  if (output !== undefined && !['lines', 'words', 'spatial'].includes(output as string))
    throw new TypeError('OcrRunOptions.output is invalid.');
  for (const key of [
    'autoUnwarp',
    'layoutReading',
    'orientationDetection',
    'lineClassification',
  ] as const) {
    const value = options[key];
    if (value !== undefined && typeof value !== 'boolean')
      throw new TypeError(`OcrRunOptions.${key} must be a boolean.`);
  }

  if (options.layoutModel !== undefined) {
    if (typeof options.layoutModel !== 'string' || options.layoutModel.trim().length === 0)
      throw new TypeError('OcrRunOptions.layoutModel must be a non-empty string.');
  }

  normalizePreDetectionOptions(options.preDetection);
  normalizeDetectionOptions(options.detection);
  normalizePostDetectionOptions(options.postDetection);
  normalizePreRecognitionOptions(options.preRecognition);
  normalizeRecognitionOptions(options.recognition);
  normalizePostRecognitionOptions(options.postRecognition);
  return options;
}

function normalizeInitializeConfig(config: InitializeConfig | undefined): InitializeConfig {
  if (config === undefined) return {};
  if (!isObject(config)) throw new TypeError('InitializeConfig must be an object.');
  requireOnlyKeys(
    config,
    [
      'preset',
      'model',
      'modelPath',
      'lang',
      'cacheDir',
      'enableReadingOrder',
      'layoutVariant',
      'models',
      'recBatchNum',
    ],
    'InitializeConfig contains an unknown key.'
  );
  if (config.preset !== undefined && !PRESETS.includes(config.preset as ModelPreset))
    throw new TypeError('InitializeConfig.preset is invalid.');
  for (const strKey of ['model', 'modelPath', 'lang', 'cacheDir'] as const) {
    const val = config[strKey];
    if (val !== undefined && (typeof val !== 'string' || val.trim().length === 0))
      throw new TypeError(`InitializeConfig.${strKey} must be a non-empty string.`);
  }
  if (config.enableReadingOrder !== undefined && typeof config.enableReadingOrder !== 'boolean')
    throw new TypeError('InitializeConfig.enableReadingOrder must be a boolean.');
  if (
    config.layoutVariant !== undefined &&
    config.layoutVariant !== 'cdla' &&
    config.layoutVariant !== 'publaynet'
  )
    throw new TypeError('InitializeConfig.layoutVariant must be "cdla" or "publaynet".');
  if (
    config.recBatchNum !== undefined &&
    (typeof config.recBatchNum !== 'number' || !Number.isInteger(config.recBatchNum) || config.recBatchNum < 1)
  )
    throw new TypeError('InitializeConfig.recBatchNum must be an integer >= 1.');
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
 * Fluent builder for chaining text detection options and running inference in React Native.
 */
export class DetectTextBuilder<TResult = TextResult[]> {
  private _source: { uri?: string; base64?: string } | null;
  private _options: OcrRunOptions;

  constructor(source: { uri?: string; base64?: string } | null = null) {
    this._source = source;
    this._options = {};
  }

  /** Specify the image source. */
  image(source: ImageSource): this {
    this._source = normalizeSource(source);
    return this;
  }

  /** Enable or disable document unwarping. */
  withAutoUnwarp(enabled = true): this {
    this._options.autoUnwarp = enabled;
    return this;
  }

  /** Enable or disable document layout reading order analysis. */
  withLayoutReading(enabled = true): this {
    this._options.layoutReading = enabled;
    return this;
  }

  /** Specify document layout or reading direction model identifier. */
  withLayoutModel(model: string): this {
    this._options.layoutModel = model;
    return this;
  }

  /** Enable or disable document orientation detection. */
  withOrientationDetection(enabled = true): this {
    this._options.orientationDetection = enabled;
    return this;
  }

  /** Enable or disable text line orientation classification. */
  withLineClassification(enabled = true): this {
    this._options.lineClassification = enabled;
    return this;
  }

  /** Configure pre-detection options. */
  withPreDetection(options: PreDetectionOptions): this {
    this._options.preDetection = { ...this._options.preDetection, ...options };
    return this;
  }

  /** Configure detection inference options. */
  withDetection(options: DetectionRunOptions): this {
    this._options.detection = { ...this._options.detection, ...options };
    return this;
  }

  /** Configure post-detection filtering options. */
  withPostDetection(options: PostDetectionOptions): this {
    this._options.postDetection = { ...this._options.postDetection, ...options };
    return this;
  }

  /** Configure pre-recognition clustering options. */
  withPreRecognition(options: PreRecognitionOptions): this {
    this._options.preRecognition = { ...this._options.preRecognition, ...options };
    return this;
  }

  /** Configure recognition inference options. */
  withRecognition(options: RecognitionRunOptions): this {
    this._options.recognition = { ...this._options.recognition, ...options };
    return this;
  }

  /** Configure post-recognition filtering options. */
  withPostRecognition(options: PostRecognitionOptions): this {
    this._options.postRecognition = { ...this._options.postRecognition, ...options };
    return this;
  }

  /** Set output format to line-level results. */
  asLines(): DetectTextBuilder<TextResult[]> {
    this._options.output = 'lines';
    return this as unknown as DetectTextBuilder<TextResult[]>;
  }

  /** Set output format to word-level results. */
  asWords(): DetectTextBuilder<TextResult[]> {
    this._options.output = 'words';
    return this as unknown as DetectTextBuilder<TextResult[]>;
  }

  /** Set output format to 2D spatial text layout. */
  asSpatial(): DetectTextBuilder<string> {
    this._options.output = 'spatial';
    return this as unknown as DetectTextBuilder<string>;
  }

  /** Set output format to TSV (tab-separated values). */
  asTsv(): DetectTextBuilder<string> {
    this._options.output = 'tsv';
    return this as unknown as DetectTextBuilder<string>;
  }

  /** Set output granularity. */
  withOutput(granularity: 'spatial' | 'tsv'): DetectTextBuilder<string>;
  withOutput(granularity: 'lines' | 'words'): DetectTextBuilder<TextResult[]>;
  withOutput(granularity: OutputGranularity): DetectTextBuilder<TextResult[] | string>;
  withOutput(granularity: OutputGranularity): DetectTextBuilder<TextResult[] | string> {
    this._options.output = granularity;
    return this as unknown as DetectTextBuilder<TextResult[] | string>;
  }

  /** Merge multiple options. */
  withOptions(options: OcrRunOptions & { output: 'spatial' | 'tsv' }): DetectTextBuilder<string>;
  withOptions(options: OcrRunOptions & { output: 'lines' | 'words' }): DetectTextBuilder<TextResult[]>;
  withOptions(options: OcrRunOptions): this;
  withOptions(options: OcrRunOptions): DetectTextBuilder<any> {
    this._options = { ...this._options, ...options };
    return this;
  }

  /** Execute OCR detection with chained options. */
  run(source?: ImageSource): Promise<TResult> {
    const src = source !== undefined ? normalizeSource(source) : this._source;
    if (src === null || src === undefined) {
      throw new Error('Image source must be provided to detectText() or run()');
    }
    return Rusto.detectText(src, normalizeOptions(this._options)) as Promise<TResult>;
  }
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
 * Closes the native RustO OCR engine and releases resources.
 */
export function close(): Promise<void> {
  return Rusto.close ? Rusto.close() : Promise.resolve();
}

/**
 * Checks whether the native RustO OCR engine is ready for inference.
 */
export function isReady(): Promise<boolean> {
  return Rusto.isReady ? Rusto.isReady() : Promise.resolve(true);
}

/**
 * Returns the underlying RustO engine version string.
 */
export function version(): Promise<string> {
  return Rusto.version ? Rusto.version() : Promise.resolve('0.4.0');
}


/**
 * Starts fluent builder chaining for OCR text detection.
 */
export function detectText(): DetectTextBuilder;
/**
 * Starts fluent builder chaining for OCR text detection with image source.
 */
export function detectText(source: ImageSource): DetectTextBuilder;
/**
 * Runs OCR text recognition on an image source with reconstructed 2D spatial text output.
 */
export function detectText(
  source: ImageSource,
  options: OcrRunOptions & { output: 'spatial' }
): Promise<string>;
/**
 * Runs OCR text recognition on an image source with structured line or word candidates.
 */
export function detectText(
  source: ImageSource,
  options?: OcrRunOptions & { output?: 'lines' | 'words' }
): Promise<TextResult[]>;
/**
 * Runs OCR text recognition on an image source.
 */
export function detectText(
  source: ImageSource,
  options?: OcrRunOptions
): Promise<TextResult[] | string>;
export function detectText(
  source?: ImageSource,
  options?: OcrRunOptions
): DetectTextBuilder | Promise<TextResult[] | string> {
  if (arguments.length === 0) {
    return new DetectTextBuilder();
  }
  const normSource = normalizeSource(source!);
  if (options !== undefined) {
    return Rusto.detectText(normSource, normalizeOptions(options));
  }
  return new DetectTextBuilder(normSource);
}
