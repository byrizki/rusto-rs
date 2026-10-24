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

export interface DetectionOptions {
  limitSideLen?: number;
  limitType?: 'min' | 'max';
  mean?: [number, number, number];
  std?: [number, number, number];
}

export interface PostprocessOptions {
  threshold?: number;
  boxThreshold?: number;
  maxCandidates?: number;
  unclipRatio?: number;
  useDilation?: boolean;
}

export interface OcrRunOptions {
  output?: 'lines' | 'words' | 'spatial';
  lineYThreshold?: number;
  wordXThreshold?: number;
  textScore?: number;
  minHeight?: number;
  maxSideLen?: number;
  minSideLen?: number;
  widthHeightRatio?: number;
  detection?: DetectionOptions;
  postprocess?: PostprocessOptions;
}

export interface InitializeConfig {
  preset?: 'ppv6' | 'ppv5' | 'ppv4' | 'ppv3';
  models?: {
    detection?: string;
    recognition?: string;
    dictionary?: string;
    classification?: string;
    orientation?: string;
  };
}

export class RustO {
  static initialize(config: InitializeConfig | string): RustO;
  detectTextFile(path: string, options?: OcrRunOptions | string): TextResult[];
  detectTextData(data: Buffer | Uint8Array, options?: OcrRunOptions | string): TextResult[];
  detectTextSpatial(path: string, options?: OcrRunOptions | string): string;
  version(): string;
}
