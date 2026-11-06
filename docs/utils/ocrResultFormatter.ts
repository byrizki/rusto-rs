import type {
  DetectTextOptions,
  TextResult,
} from '@rustors/web';
import type { OcrModelOption } from '../types/ocrModels';
import { type ActiveRunConfig, type OcrConfig, buildDetectTextOptions } from '../types/ocrConfig';

export function processOcrOutput(
  output: TextResult[] | string,
  _lineYThreshold?: number,
  _wordXThreshold?: number,
  _enableReadingOrder: boolean = false
): { results: TextResult[]; spatialText: string } {
  if (Array.isArray(output)) {
    return {
      results: output,
      spatialText: output.map((r) => r.text).join('\n'),
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
      `${r.text.replace(/[\t\n\r]/g, ' ')}\t${r.score.toFixed(4)}\t${r.frame.left}\t${r.frame.top}\t${r.frame.width}\t${r.frame.height}`
  );
  return [header, ...rows].join('\n');
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
    `// Execute OCR with fluent builder chaining`,
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
    `// Execute OCR with structured OcrRunOptions`,
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
