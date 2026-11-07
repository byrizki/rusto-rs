import type { DetectTextOptions, TextResult } from '@rustors/web';
import type { OcrModelOption } from '../types/ocrModels';
import { type ActiveRunConfig, type OcrConfig, buildDetectTextOptions } from '../types/ocrConfig';

interface SpatialToken {
  x: number;
  y: number;
  width: number;
  height: number;
  text: string;
}

function groupSpatialLines(tokens: SpatialToken[], yTolerance: number): SpatialToken[][] {
  const sorted = tokens.toSorted((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y));
  const lines: SpatialToken[][] = [];
  let currentLine: SpatialToken[] = [];
  let currentLineYSum = 0;

  for (const token of sorted) {
    if (currentLine.length > 0) {
      const avgY = currentLineYSum / currentLine.length;
      if (Math.abs(token.y - avgY) <= yTolerance) {
        currentLine.push(token);
        currentLineYSum += token.y;
      } else {
        lines.push(currentLine);
        currentLine = [token];
        currentLineYSum = token.y;
      }
    } else {
      currentLine = [token];
      currentLineYSum = token.y;
    }
  }
  if (currentLine.length > 0) lines.push(currentLine);
  return lines;
}

function formatSpatialLine(
  line: SpatialToken[],
  minX: number,
  avgCharWidth: number,
  xGapThreshold: number
): string {
  const sortedLine = line.toSorted((a, b) => a.x - b.x);
  let lineStr = '';
  let currentCharPos = 0;
  let prevTokenEndX = minX;

  for (let i = 0; i < sortedLine.length; i++) {
    const token = sortedLine[i]!;
    const targetCharPos = (token.x - minX) / avgCharWidth;
    const spacesNeeded =
      i === 0
        ? Math.max(0, Math.round(targetCharPos))
        : token.x - prevTokenEndX < xGapThreshold
          ? 1
          : Math.max(1, Math.round(targetCharPos - currentCharPos));

    const cappedSpaces = Math.min(spacesNeeded, 80);
    lineStr += ' '.repeat(cappedSpaces) + token.text;
    currentCharPos += cappedSpaces + token.text.length;
    prevTokenEndX = token.x + token.width;
  }
  return lineStr;
}

export function reconstructSpatialText(
  items: TextResult[],
  lineYThreshold: number = 0.5,
  wordXThreshold: number = 0.4
): string {
  if (items.length === 0) return '';

  const tokens: SpatialToken[] = items.map((item) => ({
    x: item.frame.left,
    y: item.frame.top,
    width: Math.max(item.frame.width, 1),
    height: Math.max(item.frame.height, 1),
    text: item.text,
  }));

  const sortedHeights = tokens.map((t) => t.height).toSorted((a, b) => a - b);
  const medianHeight = sortedHeights[Math.floor(sortedHeights.length / 2)] ?? 12;
  const totalChars = tokens.reduce((acc, t) => acc + t.text.length, 0);
  const totalWidth = tokens.reduce((acc, t) => acc + t.width, 0);
  const avgCharWidth =
    totalChars > 0 ? Math.max(totalWidth / totalChars, 1) : Math.max(medianHeight * 0.5, 4);

  const lines = groupSpatialLines(tokens, medianHeight * lineYThreshold);
  const minX = Math.min(...tokens.map((t) => t.x));
  const outputLines: string[] = [];
  let prevLineAvgY: number | null = null;

  for (const line of lines) {
    const lineAvgY = line.reduce((acc, t) => acc + t.y, 0) / line.length;
    if (prevLineAvgY !== null && lineAvgY - prevLineAvgY > medianHeight * 1.5) {
      const blankCount = Math.min(
        Math.max(Math.round((lineAvgY - prevLineAvgY - medianHeight) / medianHeight) - 1, 0),
        4
      );
      for (let b = 0; b < blankCount; b++) outputLines.push('');
    }
    prevLineAvgY = lineAvgY;
    outputLines.push(formatSpatialLine(line, minX, avgCharWidth, avgCharWidth * wordXThreshold));
  }

  return outputLines.join('\n');
}

export function processOcrOutput(
  output: TextResult[] | string,
  lineYThreshold: number = 0.5,
  wordXThreshold: number = 0.4,
  _enableReadingOrder: boolean = false
): { results: TextResult[]; spatialText: string } {
  if (Array.isArray(output)) {
    return {
      results: output,
      spatialText: reconstructSpatialText(output, lineYThreshold, wordXThreshold),
    };
  }
  return { results: [], spatialText: output };
}

export function formatResultsTsv(
  results: Array<{
    text: string;
    score: number;
    frame: { left: number; top: number; width: number; height: number };
  }>
): string {
  const header = ['text', 'score', 'left', 'top', 'width', 'height'].join('\t');
  const rows = results.map(
    (r) =>
      `${r.text.replaceAll(/[\t\n\r]/gu, ' ')}\t${r.score.toFixed(4)}\t${Math.round(r.frame.left)}\t${Math.round(r.frame.top)}\t${Math.round(r.frame.width)}\t${Math.round(r.frame.height)}`
  );
  return [header, ...rows].join('\n');
}

export function downloadTextFile(content: string, filename: string, mimeType: string): void {
  if (typeof document === 'undefined') return;
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function formatActiveConfigJson(
  modelName: string,
  outputMode: string,
  options: unknown
): string {
  return JSON.stringify({ model: modelName, granularity: outputMode, options }, null, 2);
}

function appendBuilderStages(lines: string[], options: DetectTextOptions): void {
  if (options.autoUnwarp) lines.push(`  .withAutoUnwarp()`);
  if (options.layoutReading) lines.push(`  .withLayoutReading()`);
  if (options.orientationDetection) lines.push(`  .withOrientationDetection()`);
  if (options.lineClassification) lines.push(`  .withLineClassification()`);
  if (options.preDetection && Object.keys(options.preDetection).length > 0) {
    lines.push(`  .withPreDetection(${JSON.stringify(options.preDetection)})`);
  }
  if (options.detection) lines.push(`  .withDetection(${JSON.stringify(options.detection)})`);
  if (options.postDetection)
    lines.push(`  .withPostDetection(${JSON.stringify(options.postDetection)})`);
  if (options.preRecognition)
    lines.push(`  .withPreRecognition(${JSON.stringify(options.preRecognition)})`);
  if (options.recognition) lines.push(`  .withRecognition(${JSON.stringify(options.recognition)})`);
  if (options.postRecognition)
    lines.push(`  .withPostRecognition(${JSON.stringify(options.postRecognition)})`);
}

export function formatFluentBuilderSnippet(
  outputMode: 'lines' | 'words' | 'spatial' | 'tsv',
  options: DetectTextOptions
): string {
  const lines: string[] = [
    `import { detectText } from '@rustors/web';`,
    ``,
    `const results = await detectText()`,
    `  .image(imageSource)`,
  ];

  appendBuilderStages(lines, options);

  if (outputMode === 'words') {
    lines.push(`  .asWords()`);
  } else if (outputMode === 'spatial') {
    lines.push(`  .asSpatial()`);
  } else if (outputMode === 'tsv') {
    lines.push(`  .asTsv()`);
  } else {
    lines.push(`  .asLines()`);
  }

  lines.push(`  .run();`);
  return lines.join('\n');
}

export function formatOptionsSnippet(options: DetectTextOptions): string {
  return [
    `import { detectText } from '@rustors/web';`,
    ``,
    `const results = await detectText(imageSource, ${JSON.stringify(options, null, 2)});`,
  ].join('\n');
}

export function buildActiveConfigJson(
  arc: ActiveRunConfig | null,
  fallbackModel: string,
  fallbackMode: 'lines' | 'words',
  fallbackConfig: OcrConfig
): string {
  return formatActiveConfigJson(
    arc?.modelName ?? fallbackModel,
    arc?.outputMode ?? fallbackMode,
    arc?.options ?? buildDetectTextOptions(fallbackConfig, fallbackMode)
  );
}

export function buildActiveRunConfig(
  model: OcrModelOption,
  outputMode: 'lines' | 'words',
  durationMs: number | null,
  itemsCount: number,
  options: DetectTextOptions,
  rawConfig: OcrConfig
): ActiveRunConfig {
  return {
    modelId: model.id,
    modelName: model.name,
    modelPreset: model.preset,
    outputMode,
    timestamp: Date.now(),
    durationMs,
    itemsCount,
    options,
    rawConfig: { ...rawConfig },
  };
}

export function isRunConfigModified(
  arc: ActiveRunConfig | null,
  selectedModel: string,
  outputMode: string,
  rawConfig: OcrConfig
): boolean {
  return (
    !!arc &&
    (selectedModel !== arc.modelId ||
      outputMode !== arc.outputMode ||
      JSON.stringify(rawConfig) !== JSON.stringify(arc.rawConfig))
  );
}

export function getImageDimensions(target: EventTarget | null): {
  naturalW: number;
  naturalH: number;
  renderW: number;
  renderH: number;
} {
  const img = target as HTMLImageElement | null;
  const rect = img?.getBoundingClientRect();
  const renderW = rect && rect.width > 0 ? rect.width : img?.clientWidth || img?.naturalWidth || 1;
  const renderH =
    rect && rect.height > 0 ? rect.height : img?.clientHeight || img?.naturalHeight || 1;
  return {
    naturalW: img?.naturalWidth || 1,
    naturalH: img?.naturalHeight || 1,
    renderW,
    renderH,
  };
}
