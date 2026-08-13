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

function bytesToBase64(bytes: BinaryImageData): string {
  const data = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
  let binary = '';
  for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
  const encoder = global.btoa;
  if (typeof encoder !== 'function') throw new Error('Base64 encoding is unavailable in this React Native runtime.');
  return encoder(binary);
}
function normalizeSource(source: ImageSource): { uri?: string; base64?: string } {
  if ('bytes' in source) return { base64: bytesToBase64(source.bytes) };
  return source;
}

export function initialize(config: InitializeConfig = {}): Promise<void> { return Rusto.initialize(config); }
export function detectText(source: ImageSource, options: DetectTextOptions & { output: 'spatial' }): Promise<string>;
export function detectText(source: ImageSource, options?: DetectTextOptions & { output?: 'lines' | 'words' }): Promise<TextResult[]>;
export function detectText(source: ImageSource, options: DetectTextOptions = {}): Promise<TextResult[] | string> {
  return Rusto.detectText(normalizeSource(source), options);
}
