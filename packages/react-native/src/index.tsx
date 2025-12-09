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

export interface TextResult {
  text: string;
  score: number;
  box_points: [[number, number], [number, number], [number, number], [number, number]];
}

export interface RustoInterface {
  initialize(detModel?: string, recModel?: string, dict?: string): Promise<boolean>;
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
  detModel?: string,
  recModel?: string,
  dict?: string
): Promise<boolean> {
  return Rusto.initialize(detModel, recModel, dict);
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
  yThresholdMultiplier: number = 0.6,
  xThresholdMultiplier: number = 1.3
): Promise<string> {
  return Rusto.detectTextToSpatialText(imagePath, yThresholdMultiplier, xThresholdMultiplier);
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
  yThresholdMultiplier: number = 0.6,
  xThresholdMultiplier: number = 1.3
): Promise<string> {
  return Rusto.detectTextFromBytesToSpatialText(imageData, yThresholdMultiplier, xThresholdMultiplier);
}

export function getVersion(): Promise<string> {
  return Rusto.getVersion();
}
