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

export function getVersion(): Promise<string> {
  return Rusto.getVersion();
}
