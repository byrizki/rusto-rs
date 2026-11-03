import {
  type DetectTextOptions,
  type TextResult,
  formatSpatialText,
  groupCandidatesIntoLines,
  sortCandidatesReadingOrder,
} from '@rustors/web';
import type { OcrModelOption } from '../types/ocrModels';
import { type ActiveRunConfig, type OcrConfig, buildDetectTextOptions } from '../types/ocrConfig';

export function processOcrOutput(
  output: TextResult[] | string,
  lineYThreshold: number,
  wordXThreshold: number,
  enableReadingOrder: boolean = false
): { results: TextResult[]; spatialText: string } {
  if (Array.isArray(output)) {
    const ordered = enableReadingOrder
      ? sortCandidatesReadingOrder(output, lineYThreshold)
      : output;
    return {
      results: ordered,
      spatialText: formatSpatialText(
        groupCandidatesIntoLines(ordered, lineYThreshold),
        wordXThreshold
      ),
    };
  }
  return { results: [], spatialText: output };
}

export function formatResultsCsv(
  results: Array<{
    text: string;
    score: number;
    frame: { left: number; top: number; width: number; height: number };
  }>
): string {
  const header = 'text,score,left,top,width,height';
  const rows = results.map(
    (r) =>
      `"${r.text.replaceAll('"', '""')}",${r.score.toFixed(4)},${r.frame.left},${r.frame.top},${r.frame.width},${r.frame.height}`
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
