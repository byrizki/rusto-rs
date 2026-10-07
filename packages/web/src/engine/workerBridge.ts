import type {
  DetectTextOptions,
  InitializeConfig,
  TextResult,
} from '../types/index.js';
import type { RawImageData } from '../utils/imageUtils.js';

export interface WorkerMessageRequest {
  id: number;
  type: 'INIT' | 'DETECT' | 'TERMINATE';
  config?: InitializeConfig;
  image?: RawImageData;
  options?: DetectTextOptions;
}

export interface WorkerMessageResponse {
  id: number;
  type: 'INIT_SUCCESS' | 'DETECT_SUCCESS' | 'ERROR';
  results?: TextResult[] | string;
  error?: string;
}

export class OcrWorkerBridge {
  private worker: Worker | null = null;
  private messageCounter = 0;
  private pendingRequests = new Map<
    number,
    { resolve: (val: any) => void; reject: (err: Error) => void }
  >();

  constructor(workerInstance?: Worker) {
    if (workerInstance) {
      this.attachWorker(workerInstance);
    }
  }

  public attachWorker(worker: Worker): void {
    this.worker = worker;
    this.worker.onmessage = (event: MessageEvent<WorkerMessageResponse>) => {
      const { id, type, results, error } = event.data;
      const pending = this.pendingRequests.get(id);
      if (!pending) return;

      this.pendingRequests.delete(id);
      if (type === 'ERROR') {
        pending.reject(new Error(error || 'Unknown Web Worker error occurred.'));
      } else {
        pending.resolve(results);
      }
    };

    this.worker.onerror = (err) => {
      for (const [, { reject }] of this.pendingRequests) {
        reject(new Error(`Worker execution error: ${err.message}`));
      }
      this.pendingRequests.clear();
    };
  }

  public async initialize(config: InitializeConfig): Promise<void> {
    if (!this.worker && typeof Worker !== 'undefined') {
      try {
        if (config.workerUrl) {
          const w = new Worker(config.workerUrl, { type: 'module' });
          this.attachWorker(w);
        }
      } catch (e) {
        console.warn('Worker instantiation failed; will execute on main thread:', e);
        return;
      }
    }

    if (!this.worker) return;

    const id = ++this.messageCounter;
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.worker!.postMessage({ id, type: 'INIT', config });
    });
  }

  public async detect(
    image: RawImageData,
    options: DetectTextOptions
  ): Promise<TextResult[] | string> {
    if (!this.worker) {
      throw new Error('Worker is not running.');
    }
    const id = ++this.messageCounter;
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.worker!.postMessage({ id, type: 'DETECT', image, options });
    });
  }

  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    for (const [, { reject }] of this.pendingRequests) {
      reject(new Error('Worker terminated.'));
    }
    this.pendingRequests.clear();
  }

  public isAvailable(): boolean {
    return this.worker !== null;
  }
}
