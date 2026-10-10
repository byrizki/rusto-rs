import type { Frame } from '@rusto/web';

export type QuadPoints = [[number, number], [number, number], [number, number], [number, number]];

export function scaleFrame(frame: Frame, scaleX: number, scaleY: number): Frame {
  return {
    left: Math.round(frame.left * scaleX * 100) / 100,
    top: Math.round(frame.top * scaleY * 100) / 100,
    width: Math.round(frame.width * scaleX * 100) / 100,
    height: Math.round(frame.height * scaleY * 100) / 100,
  };
}

export function scaleQuad(quad: QuadPoints, scaleX: number, scaleY: number): QuadPoints {
  return [
    [Math.round(quad[0][0] * scaleX * 100) / 100, Math.round(quad[0][1] * scaleY * 100) / 100],
    [Math.round(quad[1][0] * scaleX * 100) / 100, Math.round(quad[1][1] * scaleY * 100) / 100],
    [Math.round(quad[2][0] * scaleX * 100) / 100, Math.round(quad[2][1] * scaleY * 100) / 100],
    [Math.round(quad[3][0] * scaleX * 100) / 100, Math.round(quad[3][1] * scaleY * 100) / 100],
  ];
}

export function calculateScale(
  naturalWidth: number,
  naturalHeight: number,
  renderWidth: number,
  renderHeight: number
): { scaleX: number; scaleY: number } {
  if (naturalWidth <= 0 || naturalHeight <= 0 || renderWidth <= 0 || renderHeight <= 0) {
    return { scaleX: 1, scaleY: 1 };
  }
  return {
    scaleX: renderWidth / naturalWidth,
    scaleY: renderHeight / naturalHeight,
  };
}
