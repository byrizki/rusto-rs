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

/**
 * Result of text detection: structured list of TextResult objects or reconstructed spatial string.
 */
export type DetectTextResult = TextResult[] | string;
