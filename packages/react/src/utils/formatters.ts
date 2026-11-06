import type { TextResult } from '@rustors/web';

/**
 * Combines all recognized text tokens into a single text block.
 */
export function resultsToText(results: TextResult[]): string {
  return results.map((r) => r.text).join('\n');
}

/**
 * Formats OCR results into standard TSV (tab-separated values) with bounding box coordinates and scores.
 */
export function resultsToTsv(results: TextResult[]): string {
  const header = ['text', 'score', 'left', 'top', 'width', 'height'].join('\t');
  const rows = results.map((r) => {
    const safeText = r.text.replace(/[\t\n\r]/g, ' ');
    return `${safeText}\t${r.score.toFixed(4)}\t${r.frame.left}\t${r.frame.top}\t${r.frame.width}\t${r.frame.height}`;
  });
  return [header, ...rows].join('\n');
}

/**
 * Pretty formats OCR results as a JSON string.
 */
export function resultsToJson(results: TextResult[], indent: number = 2): string {
  return JSON.stringify(results, null, indent);
}
