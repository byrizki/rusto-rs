import { type ModelSourceConfig, initialize } from '@rustors/web';
import { type LayoutModelId, type OcrModelOption } from '../types/ocrModels';
import {
  getLayoutModelCandidateUrls,
  getModelCandidateUrls,
  isModelSupportedByBackend,
  type InferenceBackend,
} from './ocrReleaseModels';

export interface ModelDownloadProgress {
  stageText: string;
  percent: number;
  detailText: string;
}

export interface LoadedModelAssets extends ModelSourceConfig {
  detection: Uint8Array;
  recognition: Uint8Array;
  dictionary: string;
  layout?: Uint8Array;
  [key: string]: string | Uint8Array | ArrayBuffer | undefined;
}

/* oxlint-disable eslint(no-await-in-loop) */
/* eslint-disable no-await-in-loop */
async function streamBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  total: number,
  label: string,
  stepIndex: number,
  totalSteps: number,
  range: [number, number],
  onProgress?: (progress: ModelDownloadProgress) => void
): Promise<Uint8Array> {
  let received = 0;
  const chunks: Uint8Array[] = [];
  const [minPct, maxPct] = range;
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

export async function downloadBinaryWithProgress(
  urlOrUrls: string | string[],
  label: string,
  stepIndex: number,
  totalSteps: number,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100]
): Promise<Uint8Array> {
  const urls = Array.isArray(urlOrUrls) ? urlOrUrls : [urlOrUrls];
  let lastError: Error | null = null;
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        lastError = new Error(`HTTP ${res.status} loading ${url}`);
        continue;
      }
      const total = Number(res.headers.get('content-length')) || 0;
      const reader = res.body?.getReader();
      if (!reader) {
        const ab = await res.arrayBuffer();
        return new Uint8Array(ab);
      }
      return await streamBody(
        reader,
        total,
        label,
        stepIndex,
        totalSteps,
        progressRange,
        onProgress
      );
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }
  throw lastError ?? new Error(`Failed to load binary asset for ${label}`);
}

export function parseModelDictionary(dictRaw: string): string {
  const trimmed = dictRaw.trim();
  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      let chars: string[] = [];
      if (Array.isArray(parsed.chars)) {
        chars = parsed.chars;
      } else if (parsed.vocab) {
        const v = typeof parsed.vocab === 'string' ? JSON.parse(parsed.vocab) : parsed.vocab;
        if (Array.isArray(v.chars)) chars = v.chars;
      }
      if (chars.length > 0) return chars.join('\n');
    } catch {
      // Fallback to raw text
    }
  }
  return dictRaw;
}

export async function downloadTextAsset(urlOrUrls: string | string[]): Promise<string> {
  const urls = Array.isArray(urlOrUrls) ? urlOrUrls : [urlOrUrls];
  let lastError: Error | null = null;
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        lastError = new Error(`HTTP ${res.status} loading ${url}`);
        continue;
      }
      const text = await res.text();
      return parseModelDictionary(text);
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }
  throw lastError ?? new Error('Failed to load text asset');
}

function downloadLayoutAsset(
  layoutModelId: LayoutModelId,
  baseURL: string,
  totalSteps: number,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100],
  backend: InferenceBackend = 'rusto-nn'
): Promise<Uint8Array | undefined> {
  const layoutUrls = getLayoutModelCandidateUrls(layoutModelId, baseURL, backend);
  if (layoutUrls.length === 0) return Promise.resolve();
  const label =
    layoutModelId === 'indra' ? 'Indra Neural Reading Layout' : `${layoutModelId} (layout)`;
  return downloadBinaryWithProgress(layoutUrls, label, 3, totalSteps, onProgress, progressRange);
}

async function downloadModelWeights(
  candidates: ReturnType<typeof getModelCandidateUrls>,
  modelName: string,
  totalSteps: number,
  backend: InferenceBackend,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100]
): Promise<{ detection: Uint8Array; recognition: Uint8Array }> {
  const ext = backend === 'rusto-rten' ? 'rten' : 'mnn';
  const fetchStage = (stage: 'det' | 'rec', urls: string[], step: number) =>
    downloadBinaryWithProgress(
      urls,
      `${modelName} (${stage}.${ext})`,
      step,
      totalSteps,
      onProgress,
      progressRange
    );
  const detection = await fetchStage('det', candidates.detection, 1);
  const recognition = await fetchStage('rec', candidates.recognition, 2);
  return { detection, recognition };
}

export async function fetchModelAssets(
  model: OcrModelOption,
  onProgress?: (progress: ModelDownloadProgress) => void,
  progressRange: [number, number] = [0, 100],
  baseURL: string = '/',
  layoutModelId?: LayoutModelId,
  backend: InferenceBackend = 'rusto-nn'
): Promise<LoadedModelAssets> {
  if (!isModelSupportedByBackend(model.id, backend)) {
    throw new Error(
      `${model.name} is not compatible with ${backend}. Use rusto-nn or rusto-webgpu.`
    );
  }
  const candidates = getModelCandidateUrls(model.id, baseURL, backend);
  const needsLayout =
    layoutModelId !== undefined && layoutModelId !== 'none' && layoutModelId !== 'heuristic';
  const totalSteps = needsLayout ? 3 : 2;
  const { detection, recognition } = await downloadModelWeights(
    candidates,
    model.name,
    totalSteps,
    backend,
    onProgress,
    progressRange
  );
  const dictionary = await downloadTextAsset(candidates.dictionary);
  const layout = needsLayout
    ? await downloadLayoutAsset(
        layoutModelId,
        baseURL,
        totalSteps,
        onProgress,
        progressRange,
        backend
      )
    : undefined;
  return {
    detection,
    recognition,
    dictionary,
    ...(layout ? { layout } : {}),
  };
}

export async function loadAndInitModel(
  baseURL: string,
  model: OcrModelOption,
  progressRange: [number, number] = [0, 100],
  onProgress?: (progress: ModelDownloadProgress) => void,
  layoutModelId?: LayoutModelId,
  backend: InferenceBackend = 'rusto-nn'
): Promise<void> {
  const assets = await fetchModelAssets(
    model,
    onProgress,
    progressRange,
    baseURL,
    layoutModelId,
    backend
  );
  const wasmFileName =
    backend === 'rusto-rten'
      ? 'rusto_rten_bg.wasm'
      : backend === 'rusto-webgpu'
        ? 'rusto_webgpu_bg.wasm'
        : 'rusto_nn_bg.wasm';
  await initialize({
    backend,
    preset: model.preset,
    wasmUrl: `${baseURL}wasm/${wasmFileName}?v=0.4.0`,
    models: assets,
    enableReadingOrder: layoutModelId !== undefined && layoutModelId !== 'none',
    layoutModel: layoutModelId,
  });
}
