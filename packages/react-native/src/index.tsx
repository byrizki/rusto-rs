import { NativeModules, Platform } from 'react-native';

const LINKING_ERROR =
  `The package 'react-native-rusto' doesn't seem to be linked. Make sure: \n\n` +
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

export interface Frame {
  width: number;
  height: number;
  top: number;
  left: number;
}

export interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
  frame: Frame;
}

export interface DetectionConfig {
  /** Enable/disable text detection stage (default: true) */
  enabled?: boolean;
  /** Path to text detection model (e.g. 'det.mnn') */
  modelPath?: string;
  /** Detection binarization threshold (0.0 - 1.0, default: 0.3) */
  thresh?: number;
  /** Detection box score threshold (0.0 - 1.0, default: 0.6 for v6/v4/v3, 0.5 for v5) */
  boxThresh?: number;
  /** Polygon expansion unclip ratio (default: 2.0 for v6/v5, 1.5 for v4/v3) */
  unclipRatio?: number;
  /** Max image side length limit for detection (default: 736 for v6/v5, 960 for v4/v3) */
  limitSideLen?: number;
  /** Side limit type: 'min' or 'max' (default: 'min' for v6/v5, 'max' for v4/v3) */
  limitType?: 'min' | 'max' | string;
  /** Whether to apply morphological dilation (default: true for v6/v5, false for v4/v3) */
  useDilation?: boolean;
}

export interface RecognitionConfig {
  /** Enable/disable text recognition stage (default: true) */
  enabled?: boolean;
  /** Path to text recognition model (e.g. 'rec.mnn') */
  modelPath?: string;
  /** Path to dictionary file (e.g. 'dict.txt') */
  dictPath?: string;
  /** Minimum text confidence score (0.0 - 1.0, default: 0.5) */
  scoreThresh?: number;
  /** Return word-level bounding boxes (default: false) */
  returnWordBox?: boolean;
  /** Return character-level bounding boxes (default: false) */
  returnSingleCharBox?: boolean;
}

/**
 * Line Classification Configuration (CLS)
 * NOTE: Text line orientation classifier (180° rotation) is ONLY available on PP-OCRv4 and PP-OCRv5.
 */
export interface ClassificationConfig {
  /** Enable/disable line classification (default: false). NOTE: Available ONLY on PP-OCRv4 and PP-OCRv5 */
  enabled?: boolean;
  /** Path to text line classification model (e.g. 'cls.mnn'). NOTE: Available ONLY on PP-OCRv4 and PP-OCRv5 */
  modelPath?: string;
  /** Minimum confidence threshold for classification (0.0 - 1.0, default: 0.9). NOTE: Available ONLY on PP-OCRv4 and PP-OCRv5 */
  thresh?: number;
}

export interface OrientationConfig {
  /** Enable/disable document orientation correction (default: false) */
  enabled?: boolean;
  /** Path to document orientation model (e.g. 'orient.mnn') */
  modelPath?: string;
  /** Minimum confidence threshold for document orientation (0.0 - 1.0, default: 0.5) */
  thresh?: number;
}

export interface UnwarpConfig {
  /** Enable/disable document unwarping (default: false) */
  enabled?: boolean;
  /** Path to document unwarping model (e.g. 'unwarp.mnn') */
  modelPath?: string;
}

export interface PreprocessingConfig {
  /** Minimum text box height in pixels (default: 30.0) */
  minHeight?: number;
  /** Maximum image side length for processing (default: 2000.0) */
  maxSideLen?: number;
  /** Minimum image side length for processing (default: 30.0) */
  minSideLen?: number;
  /** Enable debug images in output (default: false) */
  debugImages?: boolean;
}

export interface LayoutConfig {
  /** Y threshold multiplier for line grouping in spatial text (default: 0.5) */
  yThresholdMultiplier?: number;
  /** X threshold multiplier for word/column gap separation in spatial text (default: 0.4) */
  xThresholdMultiplier?: number;
}

export interface RustOConfig {
  /** Template / model preset architecture: 'ppv6', 'ppv5', 'ppv4', or 'ppv3' (default: 'ppv6') */
  template?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3' | string;
  /** Text detection (DET) stage configuration */
  detection?: DetectionConfig;
  /** Text recognition (REC) stage configuration */
  recognition?: RecognitionConfig;
  /** Text line orientation classification (CLS) configuration. NOTE: Available ONLY on PP-OCRv4 and PP-OCRv5 */
  classification?: ClassificationConfig;
  /** Document orientation (ORIENT) configuration */
  orientation?: OrientationConfig;
  /** Document unwarping (UNWARP) configuration */
  unwarp?: UnwarpConfig;
  /** Preprocessing & image constraints configuration */
  preprocessing?: PreprocessingConfig;
  /** Spatial layout formatting configuration */
  layout?: LayoutConfig;
}

export interface RustoInterface {
  initialize(config?: RustOConfig): Promise<boolean>;
  detectText(imagePath: string): Promise<TextResult[]>;
  detectTextFromBytes(imageData: string): Promise<TextResult[]>;
  detectTextToRaw(imagePath: string): Promise<string>;
  detectTextToCsv(imagePath: string): Promise<string>;
  detectTextToTextWithPosition(imagePath: string): Promise<string>;
  detectTextToSpatialText(
    imagePath: string,
    yThresholdMultiplier?: number,
    xThresholdMultiplier?: number
  ): Promise<string>;
  detectTextFromBytesToRaw(imageData: string): Promise<string>;
  detectTextFromBytesToCsv(imageData: string): Promise<string>;
  detectTextFromBytesToTextWithPosition(imageData: string): Promise<string>;
  detectTextFromBytesToSpatialText(
    imageData: string,
    yThresholdMultiplier?: number,
    xThresholdMultiplier?: number
  ): Promise<string>;
  getVersion(): Promise<string>;
}

export function initialize(config?: RustOConfig): Promise<boolean> {
  return Rusto.initialize(config ?? {});
}

export function detectText(imagePath: string): Promise<TextResult[]> {
  return Rusto.detectText(imagePath);
}

export function detectTextFromBytes(imageData: string): Promise<TextResult[]> {
  return Rusto.detectTextFromBytes(imageData);
}

export function detectTextToRaw(imagePath: string): Promise<string> {
  return Rusto.detectTextToRaw(imagePath);
}

export function detectTextToCsv(imagePath: string): Promise<string> {
  return Rusto.detectTextToCsv(imagePath);
}

export function detectTextToTextWithPosition(imagePath: string): Promise<string> {
  return Rusto.detectTextToTextWithPosition(imagePath);
}

export function detectTextToSpatialText(
  imagePath: string,
  yThresholdMultiplier: number = 0.5,
  xThresholdMultiplier: number = 0.4
): Promise<string> {
  return Rusto.detectTextToSpatialText(
    imagePath,
    yThresholdMultiplier,
    xThresholdMultiplier
  );
}

export function detectTextFromBytesToRaw(imageData: string): Promise<string> {
  return Rusto.detectTextFromBytesToRaw(imageData);
}

export function detectTextFromBytesToCsv(imageData: string): Promise<string> {
  return Rusto.detectTextFromBytesToCsv(imageData);
}

export function detectTextFromBytesToTextWithPosition(imageData: string): Promise<string> {
  return Rusto.detectTextFromBytesToTextWithPosition(imageData);
}

export function detectTextFromBytesToSpatialText(
  imageData: string,
  yThresholdMultiplier: number = 0.5,
  xThresholdMultiplier: number = 0.4
): Promise<string> {
  return Rusto.detectTextFromBytesToSpatialText(
    imageData,
    yThresholdMultiplier,
    xThresholdMultiplier
  );
}

export function getVersion(): Promise<string> {
  return Rusto.getVersion();
}

