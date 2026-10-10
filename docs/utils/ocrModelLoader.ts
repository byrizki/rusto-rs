import {
  type ModelSourceConfig,
  type DetectTextOptions,
  type TextResult,
  formatSpatialText,
  groupCandidatesIntoLines,
  initialize,
} from '@rustors/web';
import {
  type OcrModelOption,
  type ActiveRunConfig,
  type OcrConfig,
  buildDetectTextOptions,
} from '../types/ocrConfig';

export interface ModelDownloadProgress {
  stageText: string;
  percent: number;
  detailText: string;
}

export interface LoadedModelAssets extends ModelSourceConfig {
  detection: Uint8Array;
  recognition: Uint8Array;
  dictionary: string;
  [key: string]: string | Uint8Array | ArrayBuffer | undefined;
}

export function processOcrOutput(
  output: TextResult[] | string,
  lineYThreshold: number,
  wordXThreshold: number
): { results: TextResult[]; spatialText: string } {
  if (Array.isArray(output)) {
    return {
      results: output,
      spatialText: formatSpatialText(
        groupCandidatesIntoLines(output, lineYThreshold),
        wordXThreshold
      ),
    };
  }
  return { results: [], spatialText: output };
}

/* oxlint-disable eslint(no-await-in-loop) */
/* eslint-disable no-await-in-loop */
export async function downloadBinaryWithProgress(
  url: string,
  label: string,
  stepIndex: number,
  totalSteps: number,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100]
): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body?.getReader();
  if (!reader) {
    const ab = await res.arrayBuffer();
    return new Uint8Array(ab);
  }
  let received = 0;
  const chunks: Uint8Array[] = [];
  const [minPct, maxPct] = progressRange;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const fraction = total > 0 ? received / total : 0.5;
    const norm = (stepIndex - 1 + fraction) / totalSteps;
    const overallPct = Math.round(minPct + norm * (maxPct - minPct));
    const mbRec = (received / (1024 * 1024)).toFixed(1);
    const mbTot = total > 0 ? `${(total / (1024 * 1024)).toFixed(1)} MB` : '...';
    onProgress?.({
      stageText: `Downloading ${label}`,
      percent: Math.min(overallPct, maxPct),
      detailText: `${mbRec} / ${mbTot} (${overallPct}%)`,
    });
  }
  const output = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    output.set(c, offset);
    offset += c.length;
  }
  return output;
}

export async function downloadTextAsset(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  return await res.text();
}

export async function fetchModelAssets(
  baseURL: string,
  model: OcrModelOption,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100]
): Promise<LoadedModelAssets> {
  const detUrl = `${baseURL}models/${model.id}/det.rten`;
  const recUrl = `${baseURL}models/${model.id}/rec.rten`;
  const dictUrl = `${baseURL}models/${model.id}/dict.txt`;

  const detBuffer = await downloadBinaryWithProgress(
    detUrl,
    `${model.name} (det.rten)`,
    1,
    2,
    onProgress,
    progressRange
  );
  const recBuffer = await downloadBinaryWithProgress(
    recUrl,
    `${model.name} (rec.rten)`,
    2,
    2,
    onProgress,
    progressRange
  );
  const dictText = await downloadTextAsset(dictUrl);

  return {
    detection: detBuffer,
    recognition: recBuffer,
    dictionary: dictText,
  };
}

export async function loadAndInitModel(
  baseURL: string,
  model: OcrModelOption,
  progressRange: [number, number] = [0, 100],
  onProgress?: (progress: ModelDownloadProgress) => void
): Promise<void> {
  const assets = await fetchModelAssets(baseURL, model, onProgress, progressRange);
  await initialize({
    preset: model.preset,
    wasmUrl: `${baseURL}wasm/rusto_rten_wasm_bg.wasm`,
    models: assets,
  });
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
  return {
    naturalW: img?.naturalWidth || 1,
    naturalH: img?.naturalHeight || 1,
    renderW: img?.clientWidth || img?.naturalWidth || 1,
    renderH: img?.clientHeight || img?.naturalHeight || 1,
  };
}
