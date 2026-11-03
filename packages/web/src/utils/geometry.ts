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
 * Orders candidate text lines into natural human reading order across multi-column layouts.
 *
 * Symmetrical with Rust core `src/layout.rs`:
 * 1. Partitions page into vertical sections bounded by full-width spanning lines (>65% page width).
 * 2. Groups candidates within each section into horizontal columns based on x-bounds.
 * 3. Sorts lines within each column top-to-bottom.
 * 4. Orders columns left-to-right.
 */
export function sortCandidatesReadingOrder(
  candidates: TextResult[],
  lineYThreshold: number = 0.5,
  pageWidth?: number
): TextResult[] {
  if (candidates.length <= 1) return [...candidates];

  const pw =
    pageWidth || Math.max(...candidates.map((c) => c.frame.left + c.frame.width));
  const heights = candidates.map((c) => c.frame.height).sort((a, b) => a - b);
  const medianH = Math.max(1, heights[Math.floor(heights.length / 2)] || 20);
  const yTolerance = medianH * lineYThreshold;

  // Sort initially top-to-bottom, left-to-right
  const sorted = [...candidates].sort((a, b) => {
    if (Math.abs(a.frame.top - b.frame.top) > yTolerance) {
      return a.frame.top - b.frame.top;
    }
    return a.frame.left - b.frame.left;
  });

  // Partition into vertical sections bounded by full-width spanning headlines (>55% page width)
  const sections: TextResult[][] = [];
  let currentSection: TextResult[] = [];

  for (const item of sorted) {
    const isSpanning = item.frame.width > 0.55 * pw;
    if (isSpanning) {
      if (currentSection.length > 0) {
        sections.push(currentSection);
        currentSection = [];
      }
      sections.push([item]);
    } else {
      currentSection.push(item);
    }
  }
  if (currentSection.length > 0) {
    sections.push(currentSection);
  }

  const ordered: TextResult[] = [];

  for (const section of sections) {
    if (section.length <= 1) {
      ordered.push(...section);
      continue;
    }

    // Group items into columns
    const columns: TextResult[][] = [];
    for (const item of section) {
      const itemX1 = item.frame.left;
      const itemX2 = item.frame.left + item.frame.width;

      let bestCol = -1;
      let bestOverlap = 0;

      for (let c = 0; c < columns.length; c++) {
        const col = columns[c]!;
        const colX1 = Math.min(...col.map((r) => r.frame.left));
        const colX2 = Math.max(...col.map((r) => r.frame.left + r.frame.width));

        const overlapX1 = Math.max(itemX1, colX1);
        const overlapX2 = Math.min(itemX2, colX2);
        const overlap = Math.max(0, overlapX2 - overlapX1);

        if (overlap > bestOverlap) {
          bestOverlap = overlap;
          bestCol = c;
        }
      }

      if (bestCol >= 0 && bestOverlap > Math.min(item.frame.width, 40) * 0.3) {
        columns[bestCol]!.push(item);
      } else {
        columns.push([item]);
      }
    }

    // Merge columns that significantly overlap horizontally (>35% of either width)
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < columns.length; i++) {
        for (let j = i + 1; j < columns.length; j++) {
          const colA = columns[i]!;
          const colB = columns[j]!;
          const aX1 = Math.min(...colA.map((r) => r.frame.left));
          const aX2 = Math.max(...colA.map((r) => r.frame.left + r.frame.width));
          const bX1 = Math.min(...colB.map((r) => r.frame.left));
          const bX2 = Math.max(...colB.map((r) => r.frame.left + r.frame.width));
          const aW = Math.max(1, aX2 - aX1);
          const bW = Math.max(1, bX2 - bX1);
          const overlap = Math.max(0, Math.min(aX2, bX2) - Math.max(aX1, bX1));
          if (overlap / Math.min(aW, bW) > 0.35) {
            colA.push(...colB);
            columns.splice(j, 1);
            merged = true;
            break;
          }
        }
        if (merged) break;
      }
    }

    // Sort columns left-to-right by their min left coordinate
    columns.sort((colA, colB) => {
      const minA = Math.min(...colA.map((r) => r.frame.left));
      const minB = Math.min(...colB.map((r) => r.frame.left));
      return minA - minB;
    });

    // Within each column, sort top-to-bottom
    for (const col of columns) {
      col.sort((a, b) => a.frame.top - b.frame.top);
      ordered.push(...col);
    }
  }

  return ordered;
}

/**
 * Clusters adjacent items on the same horizontal line into merged line segments.
 * Only merges adjacent items whose horizontal gap is within maxGap (avgHeight * wordXThreshold).
 * Items separated by gaps greater than maxGap remain distinct candidates (e.g. separate columns).
 */
export function clusterLineSegments(
  line: TextResult[],
  wordXThreshold: number = 0.4
): TextResult[] {
  if (line.length <= 1) return line;

  const segments: TextResult[] = [];
  let currentSegment: TextResult | null = null;

  for (const item of line) {
    if (!currentSegment) {
      currentSegment = { ...item };
      continue;
    }

    const prevRight = currentSegment.frame.left + currentSegment.frame.width;
    const gap = item.frame.left - prevRight;
    const avgHeight = (currentSegment.frame.height + item.frame.height) / 2;
    const maxGap = Math.max(1, avgHeight * wordXThreshold);

    if (gap <= maxGap) {
      const text = `${currentSegment.text} ${item.text}`.trim();
      const avgScore: number = (currentSegment.score + item.score) / 2;
      const minX = Math.min(currentSegment.frame.left, item.frame.left);
      const minY = Math.min(currentSegment.frame.top, item.frame.top);
      const maxX = Math.max(prevRight, item.frame.left + item.frame.width);
      const maxY = Math.max(
        currentSegment.frame.top + currentSegment.frame.height,
        item.frame.top + item.frame.height
      );
      const quad: QuadPoints = [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
      ];
      currentSegment = {
        text,
        score: Math.round(avgScore * 1000) / 1000,
        box_points: quad,
        frame: calculateFrameFromQuad(quad),
      };
    } else {
      segments.push(currentSegment);
      currentSegment = { ...item };
    }
  }

  if (currentSegment) {
    segments.push(currentSegment);
  }

  return segments;
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
