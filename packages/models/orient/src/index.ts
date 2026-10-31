/**
 * RustO Pre-trained Model Package: Document Orientation Classifier
 * Package: @rustors/model-orient
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
  classificationRten?: string;
  classificationOnnx?: string;
  orientation?: string;
  orientationRten?: string;
  orientationOnnx?: string;
  unwarp?: string;
  unwarpRten?: string;
  unwarpOnnx?: string;
  layout?: string;
  layoutRten?: string;
  layoutOnnx?: string;
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
  name: '@rustors/model-orient',
  version: '0.3.2',
  title: 'Document Orientation Classifier',
  preset: 'pipeline',
  tier: 'optional',
  language: 'multilingual',
  files: {
    "orientationRten": "models/orient.rten",
    "orientation": "models/orient.rten",
    "orientationOnnx": "models/orient.onnx"
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
