import { MODEL_RELEASE_ASSETS } from './models/ocrModelAssets';

export { type ModelAssetFiles, MODEL_RELEASE_ASSETS } from './models/ocrModelAssets';

export interface HfModelAssetInfo {
  path: string;
  name: string;
  size: number;
  type: 'file' | 'directory';
  downloadUrl: string;
}

export const HUGGINGFACE_REPO = 'byrizki/rten-models';
export const HUGGINGFACE_BASE = 'https://huggingface.co/byrizki/rten-models/resolve/main';
export const HUGGINGFACE_API_BASE = 'https://huggingface.co/api/models/byrizki/rten-models';
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

export function getModelCandidateUrls(
  modelId: string,
  baseURL: string = '/'
): {
  detection: string[];
  recognition: string[];
  dictionary: string[];
} {
  const assetItem = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['indraocr']!;
  const cleanBase = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
  const isIndra = modelId.startsWith('indra');
  const defaultDict = isIndra ? 'meta.json' : 'dict.txt';

  return {
    detection: dedupeStrings([
      `${HUGGINGFACE_BASE}/${assetItem.detection}`,
      `${HUGGINGFACE_BASE}/${modelId}/det.rten`,
      `${cleanBase}/models/${assetItem.detection}`,
      `${cleanBase}/models/${modelId}/det.rten`,
      assetItem.legacyDetection ? `${cleanBase}/models/${assetItem.legacyDetection}` : undefined,
    ]),
    recognition: dedupeStrings([
      `${HUGGINGFACE_BASE}/${assetItem.recognition}`,
      `${HUGGINGFACE_BASE}/${modelId}/rec.rten`,
      `${cleanBase}/models/${assetItem.recognition}`,
      `${cleanBase}/models/${modelId}/rec.rten`,
      assetItem.legacyRecognition
        ? `${cleanBase}/models/${assetItem.legacyRecognition}`
        : undefined,
    ]),
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
  const assetItem = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['indraocr']!;
  return {
    detection: `${HUGGINGFACE_BASE}/${assetItem.detection}`,
    recognition: `${HUGGINGFACE_BASE}/${assetItem.recognition}`,
    dictionary: `${HUGGINGFACE_BASE}/${assetItem.dictionary}`,
  };
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
