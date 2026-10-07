import type {
  DetectTextOptions,
  InitializeConfig,
  ImageSource,
  TextResult,
} from '../types/index.js';
import {
  normalizeInitializeConfig,
  normalizeOptions,
  validateImageSource,
} from '../utils/validator.js';
import { extractImageData, type RawImageData } from '../utils/imageUtils.js';
import {
  calculateFrameFromQuad,
  formatSpatialText,
  groupCandidatesIntoLines,
  type QuadPoints,
} from '../utils/geometry.js';
import {
  loadModelAssets,
  initWasmRuntime,
  RtenOcrPipeline,
  type LoadedModels,
} from './wasmLoader.js';
import { OcrWorkerBridge } from './workerBridge.js';

export class WebOcrEngine {
  private initialized = false;
  private currentConfig: InitializeConfig = {};
  private loadedModels: LoadedModels = {};
  private pipeline: RtenOcrPipeline | null = null;
  private workerBridge: OcrWorkerBridge = new OcrWorkerBridge();

  /**
   * Initializes the Web OCR engine with RTen WebAssembly runtime and models.
   */
  public async initialize(config: InitializeConfig = {}): Promise<void> {
    const validConfig = normalizeInitializeConfig(config);
    this.currentConfig = { ...validConfig };

    // 1. Initialize WebAssembly RTen runtime
    await initWasmRuntime(validConfig.wasmUrl);

    // 2. Load model buffers if provided
    if (validConfig.models) {
      this.loadedModels = await loadModelAssets(validConfig.models);
    }

    // 3. Construct RTen OCR Pipeline if models and dictionary are present
    if (
      this.loadedModels.detection &&
      this.loadedModels.recognition &&
      this.loadedModels.dictionary
    ) {
      this.pipeline = new RtenOcrPipeline(
        this.loadedModels.detection,
        this.loadedModels.recognition,
        this.loadedModels.dictionary
      );
    }

    // 4. Initialize worker bridge if enabled and in browser environment
    if (validConfig.worker && typeof Worker !== 'undefined') {
      try {
        await this.workerBridge.initialize(validConfig);
      } catch (err) {
        console.warn('Worker initialization failed; falling back to main-thread RTen engine:', err);
      }
    }

    this.initialized = true;
  }

  /**
   * Checks if the engine has been successfully initialized.
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Runs text detection and recognition on an image source.
   */
  public async detectText(
    source: ImageSource,
    options: DetectTextOptions & { output: 'spatial' }
  ): Promise<string>;
  public async detectText(
    source: ImageSource,
    options?: DetectTextOptions & { output?: 'lines' | 'words' }
  ): Promise<TextResult[]>;
  public async detectText(
    source: ImageSource,
    options?: DetectTextOptions
  ): Promise<TextResult[] | string>;
  public async detectText(
    source: ImageSource,
    options: DetectTextOptions = {}
  ): Promise<TextResult[] | string> {
    if (!this.initialized) {
      throw new Error(
        'Rusto Web engine is not initialized. Call initialize() before calling detectText().'
      );
    }

    validateImageSource(source);
    const validOptions = normalizeOptions(options);

    // 1. Extract raw pixel data and dimensions
    const rawImage = await extractImageData(source);

    // 2. If worker bridge is active, delegate execution to worker thread (prevents UI freeze)
    if (this.workerBridge.isAvailable()) {
      return await this.workerBridge.detect(rawImage, validOptions);
    }

    // 3. Main-thread execution with event-loop yield
    return await this.executePipeline(rawImage, validOptions);
  }

  /**
   * Pipeline execution running RTen model inference and post-processing.
   * Yields to the event loop so that animations/render frames remain fluid.
   */
  private async executePipeline(
    image: RawImageData,
    options: DetectTextOptions
  ): Promise<TextResult[] | string> {
    const outputMode = options.output ?? 'lines';
    const lineYThreshold = options.lineYThreshold ?? 0.5;
    const wordXThreshold = options.wordXThreshold ?? 0.4;
    const minScore = options.textScore ?? 0.0;

    let rawCandidates: TextResult[] = [];

    if (this.pipeline) {
      // Yield to the event loop once before heavy computation to allow the browser to paint loading states
      await new Promise<void>((resolve) => {
        if (typeof setTimeout === 'function') {
          setTimeout(resolve, 0);
        } else {
          resolve();
        }
      });

      const detThresh = options.postprocess?.threshold ?? 0.3;
      const boxThresh = options.postprocess?.boxThreshold ?? 0.5;
      const unclipRatio = options.postprocess?.unclipRatio ?? 1.6;
      const applyCalibration =
        options.calibration?.enabled ?? this.currentConfig.calibration?.enabled ?? true;

      const pixelBytes =
        image.data instanceof Uint8Array
          ? image.data
          : new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);

      const jsonStr = this.pipeline.detectAndRecognize(
        pixelBytes,
        image.width,
        image.height,
        detThresh,
        boxThresh,
        unclipRatio,
        applyCalibration
      );

      try {
        const parsed = JSON.parse(jsonStr) as Array<{
          text: string;
          score: number;
          box_points: [[number, number], [number, number], [number, number], [number, number]];
          frame: { left: number; top: number; width: number; height: number };
        }>;
        rawCandidates = parsed.map((item) => ({
          text: item.text,
          score: item.score,
          box_points: item.box_points as QuadPoints,
          frame: { ...item.frame },
        }));
      } catch (err) {
        console.error('Failed to parse RTen OCR results:', err);
        rawCandidates = [];
      }
    }

    // Filter by score threshold
    const filteredCandidates = rawCandidates.filter((item) => item.score >= minScore);

    // Group into rows based on vertical center tolerance
    const groupedLines = groupCandidatesIntoLines(filteredCandidates, lineYThreshold);

    if (outputMode === 'spatial') {
      return formatSpatialText(groupedLines, wordXThreshold);
    }

    if (outputMode === 'words') {
      const words: TextResult[] = [];
      for (const line of groupedLines) {
        for (const item of line) {
          const parts = item.text.split(/\s+/).filter(Boolean);
          if (parts.length <= 1) {
            words.push(item);
          } else {
            const charWidth = item.frame.width / Math.max(1, item.text.length);
            let currentLeft = item.frame.left;
            for (const part of parts) {
              const partWidth = Math.max(1, part.length * charWidth);
              const partQuad: QuadPoints = [
                [currentLeft, item.frame.top],
                [currentLeft + partWidth, item.frame.top],
                [currentLeft + partWidth, item.frame.top + item.frame.height],
                [currentLeft, item.frame.top + item.frame.height],
              ];
              words.push({
                text: part,
                score: item.score,
                box_points: partQuad,
                frame: calculateFrameFromQuad(partQuad),
              });
              currentLeft += partWidth + charWidth;
            }
          }
        }
      }
      return words;
    }

    // Default 'lines': return line-level candidates
    const flatLines: TextResult[] = [];
    for (const line of groupedLines) {
      if (line.length === 1) {
        flatLines.push(line[0]);
      } else if (line.length > 1) {
        const text = line.map((item) => item.text).join(' ');
        const avgScore = line.reduce((sum, item) => sum + item.score, 0) / line.length;
        const minX = Math.min(...line.map((item) => item.frame.left));
        const minY = Math.min(...line.map((item) => item.frame.top));
        const maxX = Math.max(...line.map((item) => item.frame.left + item.frame.width));
        const maxY = Math.max(...line.map((item) => item.frame.top + item.frame.height));
        const quad: QuadPoints = [
          [minX, minY],
          [maxX, minY],
          [maxX, maxY],
          [minX, maxY],
        ];
        flatLines.push({
          text,
          score: Math.round(avgScore * 1000) / 1000,
          box_points: quad,
          frame: calculateFrameFromQuad(quad),
        });
      }
    }

    return flatLines;
  }

  /**
   * Terminates active workers and cleans up engine resources.
   */
  public terminate(): void {
    this.workerBridge.terminate();
    if (this.pipeline) {
      this.pipeline.free();
      this.pipeline = null;
    }
    this.initialized = false;
    this.loadedModels = {};
  }
}
