import type { InferenceBackend, ModelSourceConfig } from '../types/index.js';
import initRustoNn, {
  RtenOcrPipeline as NnPipeline,
  RtenModel as NnModel,
  engineBackend as nnEngineBackend,
  version as nnVersion,
  isReady as nnIsReady,
} from '../wasm/rusto_nn.js';
import initRustoRten, {
  RtenOcrPipeline as RtenPipeline,
  RtenModel as RtenModelClass,
  engineBackend as rtenEngineBackend,
  version as rtenVersion,
  isReady as rtenIsReady,
} from '../wasm/rusto_rten.js';
import initRustoWebGpu, {
  RtenOcrPipeline as WebGpuPipeline,
  RtenModel as WebGpuModel,
  initWebGpu as webgpuInitAsync,
  engineBackend as webgpuEngineBackend,
  version as webgpuVersion,
  isReady as webgpuIsReady,
} from '../wasm/rusto_webgpu.js';

export interface LoadedModels {
  detection?: Uint8Array;
  recognition?: Uint8Array;
  dictionary?: string;
  classification?: Uint8Array;
  orientation?: Uint8Array;
  layout?: Uint8Array;
}

let activeBackend: InferenceBackend = 'rusto-nn';
let activePipelineClass: any = NnPipeline;
let activeModelClass: any = NnModel;
const initializedBackends = new Set<InferenceBackend>();

/**
 * Initializes the requested WebAssembly inference runtime engine.
 * - 'rusto-nn' (default): 100% pure-Rust Alibaba MNN/ONNX execution graph.
 * - 'rusto-rten': RTen pure-Rust neural network inference engine.
 * - 'rusto-webgpu': WebGPU hardware accelerated compute shader engine.
 */
export async function initWasmRuntime(
  wasmSource?: string | ArrayBuffer | Uint8Array,
  backend: InferenceBackend = 'rusto-nn'
): Promise<void> {
  activeBackend = backend;

  if (backend === 'rusto-webgpu') {
    if (!initializedBackends.has('rusto-webgpu')) {
      if (wasmSource) {
        await initRustoWebGpu(wasmSource);
      } else {
        await initRustoWebGpu();
      }
      const hasWebGpuInBrowser =
        typeof navigator !== 'undefined' &&
        'gpu' in navigator &&
        Boolean((navigator as { gpu?: unknown }).gpu);

      if (hasWebGpuInBrowser) {
        try {
          await webgpuInitAsync();
        } catch (err) {
          console.warn('WebGPU compute initialization failed; falling back to CPU execution:', err);
        }
      }
      initializedBackends.add('rusto-webgpu');
    }
    activePipelineClass = WebGpuPipeline;
    activeModelClass = WebGpuModel;
    return;
  }

  if (backend === 'rusto-rten') {
    if (!initializedBackends.has('rusto-rten')) {
      if (wasmSource) {
        await initRustoRten(wasmSource);
      } else {
        await initRustoRten();
      }
      initializedBackends.add('rusto-rten');
    }
    activePipelineClass = RtenPipeline;
    activeModelClass = RtenModelClass;
    return;
  }

  // Default: rusto-nn
  if (!initializedBackends.has('rusto-nn')) {
    if (wasmSource) {
      await initRustoNn(wasmSource);
    } else {
      await initRustoNn();
    }
    initializedBackends.add('rusto-nn');
  }
  activePipelineClass = NnPipeline;
  activeModelClass = NnModel;
}

/**
 * Returns whether WebAssembly engine runtime has completed initialization.
 */
export function isWasmRuntimeInitialized(): boolean {
  return initializedBackends.has(activeBackend);
}

/**
 * Returns the active inference engine backend identifier string.
 */
export function getActiveEngineBackend(): string {
  if (activeBackend === 'rusto-webgpu') {
    return typeof webgpuEngineBackend === 'function' ? webgpuEngineBackend() : 'rusto-webgpu';
  }
  if (activeBackend === 'rusto-rten') {
    return typeof rtenEngineBackend === 'function' ? rtenEngineBackend() : 'rusto-rten';
  }
  return typeof nnEngineBackend === 'function' ? nnEngineBackend() : 'rusto-nn';
}

/**
 * Dynamic proxy class wrapping the active backend's OCR pipeline.
 */
export class RtenOcrPipeline {
  private _inner: any;

  constructor(
    detection: Uint8Array,
    recognition: Uint8Array,
    dictionary: string,
    layout?: Uint8Array
  ) {
    if (!activePipelineClass) {
      throw new Error('WebAssembly runtime not initialized. Call initialize() first.');
    }
    if (typeof activePipelineClass.newWithModels === 'function' && layout) {
      this._inner = activePipelineClass.newWithModels(
        detection,
        recognition,
        dictionary,
        layout
      );
    } else {
      this._inner = new activePipelineClass(detection, recognition, dictionary);
      if (layout && typeof this._inner.setLayoutModel === 'function') {
        this._inner.setLayoutModel(layout);
      }
    }
  }

  setLayoutModel(layout_data: Uint8Array): void {
    if (this._inner && typeof this._inner.setLayoutModel === 'function') {
      this._inner.setLayoutModel(layout_data);
    }
  }

  detectText(
    image_bytes: Uint8Array,
    width: number,
    height: number,
    channels: number,
    stride?: number | null,
    options_json?: string | null
  ): string {
    return this._inner.detectText(image_bytes, width, height, channels, stride, options_json);
  }

  detectTextRgb(
    image_rgb: Uint8Array,
    width: number,
    height: number,
    stride?: number | null,
    options_json?: string | null
  ): string {
    return this._inner.detectTextRgb(image_rgb, width, height, stride, options_json);
  }

  detectAndRecognize(
    image_bytes: Uint8Array,
    width: number,
    height: number,
    channels: number,
    stride?: number | null,
    options_json?: string | null
  ): string {
    return this._inner.detectAndRecognize(
      image_bytes,
      width,
      height,
      channels,
      stride,
      options_json
    );
  }

  isReady(): boolean {
    return this._inner ? this._inner.is_ready() : false;
  }

  version(): string {
    return this._inner ? this._inner.version() : '';
  }

  free(): void {
    if (this._inner && typeof this._inner.free === 'function') {
      this._inner.free();
    }
  }
}

/**
 * Dynamic proxy class wrapping the active backend's tensor model.
 */
export class RtenModel {
  private _inner: any;

  constructor(model_data: Uint8Array) {
    if (!activeModelClass) {
      throw new Error('WebAssembly runtime not initialized. Call initialize() first.');
    }
    this._inner = new activeModelClass(model_data);
  }

  findNode(name: string): number | undefined {
    return this._inner.findNode(name);
  }

  inputIds(): Uint32Array {
    return this._inner.inputIds();
  }

  outputIds(): Uint32Array {
    return this._inner.outputIds();
  }

  run(input_shape: Uint32Array, input_data: Float32Array): any {
    return this._inner.run(input_shape, input_data);
  }

  runWithIds(
    input_id: number,
    input_shape: Uint32Array,
    input_data: Float32Array,
    output_id: number
  ): any {
    return this._inner.runWithIds(input_id, input_shape, input_data, output_id);
  }

  runOrder(num_items: number, feats_data: Float32Array): any {
    return this._inner.runOrder(num_items, feats_data);
  }

  free(): void {
    if (this._inner && typeof this._inner.free === 'function') {
      this._inner.free();
    }
  }
}

const disposeSymbol = (Symbol as unknown as { readonly dispose?: symbol }).dispose;
if (disposeSymbol) {
  (RtenOcrPipeline.prototype as any)[disposeSymbol] = function (this: RtenOcrPipeline) {
    this.free();
  };
  (RtenModel.prototype as any)[disposeSymbol] = function (this: RtenModel) {
    this.free();
  };
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

  const [det, rec, dict, cls, orient, layout] = await Promise.all([
    resolveBinaryBuffer(modelsConfig.detection, 'detection'),
    resolveBinaryBuffer(modelsConfig.recognition, 'recognition'),
    resolveTextContent(modelsConfig.dictionary, 'dictionary'),
    resolveBinaryBuffer(modelsConfig.classification, 'classification'),
    resolveBinaryBuffer(modelsConfig.orientation, 'orientation'),
    resolveBinaryBuffer(modelsConfig.layout, 'layout'),
  ]);

  return {
    detection: det,
    recognition: rec,
    dictionary: dict,
    classification: cls,
    orientation: orient,
    layout,
  };
}
