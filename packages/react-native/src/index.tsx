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

export interface RustOConfig {
  /** Template / model preset architecture: 'ppv6', 'ppv5', 'ppv4', or 'ppv3' (default: 'ppv6') */
  template?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3' | string;
  /** Path to text detection model (e.g. 'det.mnn') */
  detModelPath?: string;
  /** Path to text recognition model (e.g. 'rec.mnn') */
  recModelPath?: string;
  /** Path to dictionary file (e.g. 'dict.txt') */
  dictPath?: string;
  /** Path to text line orientation classification model (CLS) */
  clsModelPath?: string;
  /** Path to document orientation model */
  orientModelPath?: string;
  /** Path to text rectification / unwarp model */
  unwarpModelPath?: string;
  /** Minimum confidence threshold for document orientation (0.0 - 1.0) */
  orientThreshold?: number;
  /** Minimum confidence threshold for text line classification (0.0 - 1.0) */
  clsThreshold?: number;
  /** Minimum text confidence score (0.0 - 1.0, default: 0.5) */
  textScore?: number;
  /** Detection binarization threshold (0.0 - 1.0, default: 0.3) */
  detThresh?: number;
  /** Detection box score threshold (0.0 - 1.0, default: 0.5) */
  detBoxThresh?: number;
  /** Max image side length limit for detection (default: 736) */
  limitSideLen?: number;
  /** Side limit type: 'min' or 'max' (default: 'min') */
  limitType?: string;
  /** Polygon expansion unclip ratio (default: 2.0) */
  unclipRatio?: number;
  /** Whether to apply morphological dilation (default: true) */
  useDilation?: boolean;
  /** Enable/disable text detection (default: true) */
  useDet?: boolean;
  /** Enable/disable text recognition (default: true) */
  useRec?: boolean;
  /** Enable/disable line classification (default: false) */
  useCls?: boolean;
  /** Enable/disable document orientation correction (default: false) */
  useOrient?: boolean;
  /** Enable/disable document unwarping (default: false) */
  useUnwarp?: boolean;
  /** Enable debug images in output (default: false) */
  debugImages?: boolean;
  /** Minimum text box height in pixels (default: 30.0) */
  minHeight?: number;
  /** Maximum image side length for processing (default: 2000.0) */
  maxSideLen?: number;
  /** Minimum image side length for processing (default: 30.0) */
  minSideLen?: number;
  /** Return word-level bounding boxes (default: false) */
  returnWordBox?: boolean;
  /** Return character-level bounding boxes (default: false) */
  returnSingleCharBox?: boolean;
  /** Y threshold multiplier for line grouping in spatial text (default: 0.5) */
  yThresholdMultiplier?: number;
  /** X threshold multiplier for word/column gap separation in spatial text (default: 0.4) */
  xThresholdMultiplier?: number;
}

export interface RustoInterface {
  initialize(
    configOrDetModel?: RustOConfig | string,
    recModel?: string,
    dict?: string
  ): Promise<boolean>;
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

export function initialize(
  configOrDetModel?: RustOConfig | string,
  recModel?: string,
  dict?: string
): Promise<boolean> {
  if (typeof configOrDetModel === 'object' && configOrDetModel !== null) {
    return Rusto.initialize(configOrDetModel);
  }
  return Rusto.initialize(configOrDetModel, recModel, dict);
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

