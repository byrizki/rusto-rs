import type { Frame, TextResult } from '../types/index.js';

export type QuadPoints = [[number, number], [number, number], [number, number], [number, number]];

/**
 * Calculates axis-aligned bounding frame from 4 polygon quad vertices.
 */
export function calculateFrameFromQuad(points: QuadPoints): Frame {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  return {
    left: Math.round(minX * 100) / 100,
    top: Math.round(minY * 100) / 100,
    width: Math.round(Math.max(0, maxX - minX) * 100) / 100,
    height: Math.round(Math.max(0, maxY - minY) * 100) / 100,
  };
}

/**
 * Groups candidate boxes into lines based on vertical tolerance.
 */
export function groupCandidatesIntoLines(
  candidates: TextResult[],
  lineYThreshold: number = 0.5
): TextResult[][] {
  if (candidates.length === 0) return [];

  // Sort primarily top-to-bottom
  const sorted = [...candidates].sort((a, b) => a.frame.top - b.frame.top);
  const lines: TextResult[][] = [];

  for (const item of sorted) {
    let placed = false;
    for (const line of lines) {
      const avgLineY = line.reduce((sum, c) => sum + c.frame.top, 0) / line.length;
      const avgHeight = line.reduce((sum, c) => sum + c.frame.height, 0) / line.length;
      const tolerance = Math.max(1, avgHeight * lineYThreshold);

      if (Math.abs(item.frame.top - avgLineY) <= tolerance) {
        line.push(item);
        placed = true;
        break;
      }
    }

    if (!placed) {
      lines.push([item]);
    }
  }

  // Sort each line left-to-right
  for (const line of lines) {
    line.sort((a, b) => a.frame.left - b.frame.left);
  }

  return lines;
}

/**
 * Merges grouped line segments into formatted spatial text output.
 */
export function formatSpatialText(
  lines: TextResult[][],
  wordXThreshold: number = 0.4
): string {
  if (lines.length === 0) return '';

  let minLeft = Infinity;
  let totalWidth = 0;
  let totalChars = 0;
  let totalHeight = 0;
  let totalItems = 0;

  for (const line of lines) {
    for (const item of line) {
      if (item.text.length > 0 && item.frame.width > 0) {
        if (item.frame.left < minLeft) {
          minLeft = item.frame.left;
        }
        totalWidth += item.frame.width;
        totalChars += item.text.length;
      }
      if (item.frame.height > 0) {
        totalHeight += item.frame.height;
        totalItems += 1;
      }
    }
  }

  if (!Number.isFinite(minLeft)) {
    minLeft = 0;
  }
  const avgCharWidth = totalChars > 0 ? (totalWidth / totalChars) * (1 + wordXThreshold * 0.25) : 10;
  const avgLineHeight = totalItems > 0 ? totalHeight / totalItems : 20;

  const rowStrings: string[] = [];
  let prevLineBottom: number | null = null;

  for (const line of lines) {
    if (line.length === 0) continue;

    const lineTop = Math.min(...line.map((item) => item.frame.top));
    const lineBottom = Math.max(...line.map((item) => item.frame.top + item.frame.height));

    if (prevLineBottom !== null) {
      const verticalGap = lineTop - prevLineBottom;
      if (verticalGap > avgLineHeight * 1.5) {
        const emptyLines = Math.min(3, Math.floor(verticalGap / avgLineHeight) - 1);
        for (let i = 0; i < emptyLines; i++) {
          rowStrings.push('');
        }
      }
    }

    let rowText = '';
    let currentCol = 0;

    for (const item of line) {
      const targetCol = Math.max(0, Math.round((item.frame.left - minLeft) / avgCharWidth));
      if (targetCol > currentCol) {
        rowText += ' '.repeat(targetCol - currentCol);
        currentCol = targetCol;
      } else if (currentCol > 0 && !rowText.endsWith(' ')) {
        rowText += ' ';
        currentCol += 1;
      }
      rowText += item.text;
      currentCol = rowText.length;
    }

    rowStrings.push(rowText);
    prevLineBottom = lineBottom;
  }

  return rowStrings.join('\n');
}
