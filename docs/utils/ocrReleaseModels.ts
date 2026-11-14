import { MODEL_RELEASE_ASSETS } from './models/ocrModelAssets';

export { type ModelAssetFiles, MODEL_RELEASE_ASSETS } from './models/ocrModelAssets';

export interface HfModelAssetInfo {
  path: string;
  name: string;
  size: number;
  type: 'file' | 'directory';
  downloadUrl: string;
}

export const HUGGINGFACE_REPO = 'byrizki/rusto-models';
export const HUGGINGFACE_BASE = 'https://huggingface.co/byrizki/rusto-models/resolve/main';
export const HUGGINGFACE_API_BASE = 'https://huggingface.co/api/models/byrizki/rusto-models';
export const JSDELIVR_CDN_BASE = 'https://cdn.jsdelivr.net/npm';

function dedupeStrings(items: (string | undefined)[]): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    if (item && !seen.has(item)) {
      seen.add(item);
      result.push(item);
    }
  }
  return result;
}

export type InferenceBackend = 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';

export function isModelSupportedByBackend(modelId: string, backend: InferenceBackend): boolean {
  // RTen (.rten) supports FP32 weights only; FP16 and INT8 models require MNN/ONNX (rusto-nn or rusto-webgpu).
  if (backend === 'rusto-rten') {
    return !modelId.includes('fp16') && !modelId.includes('int8');
  }
  return true;
}

function replaceModelExtension(path: string, ext: 'mnn' | 'onnx' | 'rten'): string {
  return path.replace(/\.(mnn|onnx|rten)$/u, `.${ext}`);
}

function buildWeightCandidateUrls(
  modelId: string,
  baseKind: 'det' | 'rec',
  assetPath: string,
  legacyPath: string | undefined,
  cleanBase: string,
  backend: InferenceBackend
): string[] {
  const exts: ('mnn' | 'onnx' | 'rten')[] =
    backend === 'rusto-rten' ? ['rten'] : ['mnn', 'onnx', 'rten'];

  const hfUrls = exts.flatMap((ext) => [
    `${HUGGINGFACE_BASE}/${replaceModelExtension(assetPath, ext)}`,
    `${HUGGINGFACE_BASE}/${modelId}/${baseKind}.${ext}`,
  ]);
  const localUrls = exts.flatMap((ext) => [
    `${cleanBase}/models/${replaceModelExtension(assetPath, ext)}`,
    `${cleanBase}/models/${modelId}/${baseKind}.${ext}`,
    legacyPath ? `${cleanBase}/models/${replaceModelExtension(legacyPath, ext)}` : undefined,
  ]);

  const preferredHf = backend === 'rusto-rten' ? undefined : `${HUGGINGFACE_BASE}/${assetPath}`;
  const preferredLocal = backend === 'rusto-rten' ? undefined : `${cleanBase}/models/${assetPath}`;

  return dedupeStrings([preferredHf, ...hfUrls, preferredLocal, ...localUrls]);
}

export function getModelCandidateUrls(
  modelId: string,
  baseURL: string = '/',
  backend: InferenceBackend = 'rusto-nn'
): {
  detection: string[];
  recognition: string[];
  dictionary: string[];
} {
  const assetItem = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['ppocrv6-tiny']!;
  const cleanBase = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
  const isIndra = modelId.startsWith('indra');
  const defaultDict = isIndra ? 'meta.json' : 'dict.txt';

  return {
    detection: buildWeightCandidateUrls(
      modelId,
      'det',
      assetItem.detection,
      assetItem.legacyDetection,
      cleanBase,
      backend
    ),
    recognition: buildWeightCandidateUrls(
      modelId,
      'rec',
      assetItem.recognition,
      assetItem.legacyRecognition,
      cleanBase,
      backend
    ),
    dictionary: dedupeStrings([
      `${HUGGINGFACE_BASE}/${assetItem.dictionary}`,
      `${HUGGINGFACE_BASE}/${modelId}/${defaultDict}`,
      `${cleanBase}/models/${assetItem.dictionary}`,
      `${cleanBase}/models/${modelId}/${defaultDict}`,
      assetItem.legacyDictionary ? `${cleanBase}/models/${assetItem.legacyDictionary}` : undefined,
    ]),
  };
}

export function getModelCdnUrls(modelId: string): {
  detection: string;
  recognition: string;
  dictionary: string;
} {
  const assetItem = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['ppocrv6-tiny']!;
  return {
    detection: `${HUGGINGFACE_BASE}/${assetItem.detection}`,
    recognition: `${HUGGINGFACE_BASE}/${assetItem.recognition}`,
    dictionary: `${HUGGINGFACE_BASE}/${assetItem.dictionary}`,
  };
}

export function getLayoutModelCandidateUrls(
  layoutModelId: string,
  baseURL: string = '/',
  backend: InferenceBackend = 'rusto-nn'
): string[] {
  const cleanBase = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
  let urls: string[] = [];
  if (layoutModelId === 'indra') {
    urls = dedupeStrings([
      `${cleanBase}/models/indraocr/order.mnn`,
      `${cleanBase}/models/indraocr/order.rten`,
      `${cleanBase}/models/indraocr/order.onnx`,
      `${cleanBase}/models/indra/order.mnn`,
      `${cleanBase}/models/indra/order.rten`,
      `${cleanBase}/models/indra/order.onnx`,
      `${cleanBase}/models/indra/indra_order.rten`,
      `${cleanBase}/models/indra/indra_order.onnx`,
      `${HUGGINGFACE_BASE}/indraocr/order.mnn`,
      `${HUGGINGFACE_BASE}/indraocr/order.rten`,
      `${HUGGINGFACE_BASE}/indraocr/order.onnx`,
    ]);
  } else if (layoutModelId === 'cdla') {
    urls = dedupeStrings([
      `${HUGGINGFACE_BASE}/modules/layout_cdla.mnn`,
      `${HUGGINGFACE_BASE}/modules/layout_cdla.onnx`,
      `${HUGGINGFACE_BASE}/modules/layout_cdla.rten`,
      `${cleanBase}/models/modules/layout_cdla.mnn`,
      `${cleanBase}/models/modules/layout_cdla.onnx`,
      `${cleanBase}/models/modules/layout_cdla.rten`,
    ]);
  } else if (layoutModelId === 'publaynet') {
    urls = dedupeStrings([
      `${HUGGINGFACE_BASE}/modules/layout_publaynet.mnn`,
      `${HUGGINGFACE_BASE}/modules/layout_publaynet.onnx`,
      `${HUGGINGFACE_BASE}/modules/layout_publaynet.rten`,
      `${cleanBase}/models/modules/layout_publaynet.mnn`,
      `${cleanBase}/models/modules/layout_publaynet.onnx`,
      `${cleanBase}/models/modules/layout_publaynet.rten`,
    ]);
  }
  return backend === 'rusto-rten' ? urls.filter((url) => url.endsWith('.rten')) : urls;
}

let cachedHfIndex: HfModelAssetInfo[] | null = null;

export async function fetchHuggingFaceModelIndex(
  repo: string = HUGGINGFACE_REPO
): Promise<HfModelAssetInfo[]> {
  if (cachedHfIndex && cachedHfIndex.length > 0) {
    return cachedHfIndex;
  }
  try {
    const res = await fetch(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        cachedHfIndex = data
          .filter((item: { type: string; path: string }) => item.type === 'file')
          .map((item: { path: string; size: number }) => ({
            path: item.path,
            name: item.path.split('/').pop() || item.path,
            size: item.size || 0,
            type: 'file' as const,
            downloadUrl: `${HUGGINGFACE_BASE}/${item.path}`,
          }));
        return cachedHfIndex;
      }
    }
  } catch (err) {
    console.warn('Hugging Face model repository index query non-blocking fallback:', err);
  }
  return [];
}
