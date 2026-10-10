/**
 * RustO Pre-trained Model Package: PP-OCRv6 Small
 * Package: @rustors/model-ppocrv6-small
 */

export interface ModelFilesConfig {
  detection?: string;
  recognition?: string;
  detectionRten?: string;
  recognitionRten?: string;
  detectionOnnx?: string;
  recognitionOnnx?: string;
  dictionary?: string;
  classification?: string;
  recognitionEnglish?: string;
  dictionaryEnglish?: string;
  [key: string]: string | undefined;
}

export interface ModelPackageMetadata {
  name: string;
  version: string;
  title: string;
  preset: string;
  tier: string;
  language: string;
  files: ModelFilesConfig;
}

export const modelMetadata: ModelPackageMetadata = {
  name: '@rustors/model-ppocrv6-small',
  version: '0.3.2',
  title: 'PP-OCRv6 Small',
  preset: 'ppv6',
  tier: 'small',
  language: 'multilingual',
  files: {
    "detectionRten": "models/det.rten",
    "detection": "models/det.rten",
    "recognitionRten": "models/rec.rten",
    "recognition": "models/rec.rten",
    "dictionary": "models/dict.txt",
    "detectionOnnx": "models/det.onnx",
    "recognitionOnnx": "models/rec.onnx"
},
};

/**
 * Returns package-relative asset paths for the bundled model files.
 */
export function getModelRelativePaths(): ModelFilesConfig {
  return { ...modelMetadata.files };
}

/**
 * Resolve absolute URLs given a host base URL for browser/CDN environments.
 */
export function getModelUrls(baseUrl: string = ''): ModelFilesConfig {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const result: ModelFilesConfig = {};
  for (const [key, relativePath] of Object.entries(modelMetadata.files)) {
    if (relativePath) {
      result[key] = normalizedBase ? `${normalizedBase}/${relativePath}` : relativePath;
    }
  }
  return result;
}

/**
 * Fetch and load binary buffers in browser or Node environments.
 */
export async function loadModelBuffers(baseUrlOrPath: string = ''): Promise<Record<string, Uint8Array | string>> {
  const urls = getModelUrls(baseUrlOrPath);
  const buffers: Record<string, Uint8Array | string> = {};

  for (const [key, target] of Object.entries(urls)) {
    if (!target) continue;

    if (typeof window !== 'undefined' || typeof fetch === 'function') {
      const res = await fetch(target);
      if (!res.ok) {
        throw new Error(`Failed to fetch model asset [${key}] from ${target}: ${res.statusText}`);
      }
      if (key.includes('dict') || target.endsWith('.txt')) {
        buffers[key] = await res.text();
      } else {
        const ab = await res.arrayBuffer();
        buffers[key] = new Uint8Array(ab);
      }
    }
  }

  return buffers;
}

export default modelMetadata;
