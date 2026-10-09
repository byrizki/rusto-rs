import { WebOcrEngine } from '../engine/webEngine.js';
import type { WorkerMessageRequest, WorkerMessageResponse } from '../engine/workerBridge.js';

let engine: WebOcrEngine | null = null;

self.onmessage = async (event: MessageEvent<WorkerMessageRequest>) => {
  const { id, type, config, image, options } = event.data;

  try {
    if (type === 'INIT') {
      engine = new WebOcrEngine();
      // In worker thread, run without nested worker
      await engine.initialize({ ...config, worker: false });
      const response: WorkerMessageResponse = { id, type: 'INIT_SUCCESS' };
      self.postMessage(response);
    } else if (type === 'DETECT') {
      if (!engine || !engine.isInitialized()) {
        throw new Error('Worker OCR engine has not been initialized.');
      }
      if (!image) {
        throw new Error('Missing image data payload for text detection.');
      }

      const bytes =
        image.data instanceof Uint8Array
          ? image.data
          : new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);

      const results = await engine.detectText(
        { bytes, width: image.width, height: image.height },
        options
      );
      const response: WorkerMessageResponse = { id, type: 'DETECT_SUCCESS', results };
      self.postMessage(response);
    } else if (type === 'TERMINATE') {
      if (engine) {
        engine.terminate();
        engine = null;
      }
    }
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    const response: WorkerMessageResponse = {
      id,
      type: 'ERROR',
      error: errorMsg,
    };
    self.postMessage(response);
  }
};
