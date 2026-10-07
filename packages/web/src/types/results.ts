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

export type DetectTextResult = TextResult[] | string;
