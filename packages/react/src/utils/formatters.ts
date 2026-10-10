import type { TextResult } from '@rusto/web';

/**
 * Combines all recognized text tokens into a single text block.
 */
export function resultsToText(results: TextResult[]): string {
  return results.map((r) => r.text).join('\n');
}

/**
 * Formats OCR results into standard CSV with bounding box coordinates and scores.
 */
export function resultsToCsv(results: TextResult[]): string {
  const header = 'text,score,left,top,width,height';
  const rows = results.map((r) => {
    const escapedText = `"${r.text.replace(/"/g, '""')}"`;
    return `${escapedText},${r.score.toFixed(4)},${r.frame.left},${r.frame.top},${r.frame.width},${r.frame.height}`;
  });
  return [header, ...rows].join('\n');
}

/**
 * Pretty formats OCR results as a JSON string.
 */
export function resultsToJson(results: TextResult[], indent: number = 2): string {
  return JSON.stringify(results, null, indent);
}
