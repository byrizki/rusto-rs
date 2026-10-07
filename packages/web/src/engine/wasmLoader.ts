import type { ModelSourceConfig } from '../types/index.js';
import initWasm, { RtenOcrPipeline, RtenModel } from '../wasm/rusto_rten_wasm.js';

export interface LoadedModels {
  detection?: Uint8Array;
  recognition?: Uint8Array;
  dictionary?: string;
  classification?: Uint8Array;
  orientation?: Uint8Array;
}

let wasmReady = false;

/**
 * Initializes the RTen WebAssembly inference runtime.
 */
export async function initWasmRuntime(wasmSource?: string | ArrayBuffer | Uint8Array): Promise<void> {
  if (wasmReady) return;

  if (wasmSource) {
    await initWasm(wasmSource);
  } else {
    await initWasm();
  }
  wasmReady = true;
}

/**
 * Returns whether RTen WebAssembly engine runtime has completed initialization.
 */
export function isWasmRuntimeInitialized(): boolean {
  return wasmReady;
}

/**
 * Fetches an asset buffer from URL or passes through existing buffer.
 */
export async function resolveBinaryBuffer(
  input: string | Uint8Array | ArrayBuffer | undefined,
  assetName: string
): Promise<Uint8Array | undefined> {
  if (!input) return undefined;

  if (input instanceof Uint8Array) {
    return input;
  }

  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }

  if (typeof input === 'string') {
    if (typeof fetch !== 'function') {
      throw new Error(`fetch is unavailable to load asset [${assetName}] from "${input}".`);
    }
    const response = await fetch(input);
    if (!response.ok) {
      throw new Error(`Failed to load [${assetName}] from "${input}": ${response.statusText}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  }

  return undefined;
}

/**
 * Fetches a text asset (e.g. dictionary) from URL or passes through text/buffer.
 */
export async function resolveTextContent(
  input: string | Uint8Array | ArrayBuffer | undefined,
  _assetName: string
): Promise<string | undefined> {
  if (!input) return undefined;

  if (input instanceof Uint8Array || input instanceof ArrayBuffer) {
    const decoder = new TextDecoder('utf-8');
    return decoder.decode(input);
  }

  if (typeof input === 'string') {
    // If it contains newlines, treat directly as dictionary content
    if (input.includes('\n')) {
      return input;
    }
    if (typeof fetch === 'function') {
      try {
        const res = await fetch(input);
        if (res.ok) {
          return await res.text();
        }
      } catch {
        // Fallback: string itself might be short dictionary
      }
    }
    return input;
  }

  return undefined;
}

/**
 * Loads all configured OCR models into memory.
 */
export async function loadModelAssets(modelsConfig?: ModelSourceConfig): Promise<LoadedModels> {
  if (!modelsConfig) {
    return {};
  }

  const [det, rec, dict, cls, orient] = await Promise.all([
    resolveBinaryBuffer(modelsConfig.detection, 'detection'),
    resolveBinaryBuffer(modelsConfig.recognition, 'recognition'),
    resolveTextContent(modelsConfig.dictionary, 'dictionary'),
    resolveBinaryBuffer(modelsConfig.classification, 'classification'),
    resolveBinaryBuffer(modelsConfig.orientation, 'orientation'),
  ]);

  return {
    detection: det,
    recognition: rec,
    dictionary: dict,
    classification: cls,
    orientation: orient,
  };
}

export { RtenOcrPipeline, RtenModel };
