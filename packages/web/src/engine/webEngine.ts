import type {
  OcrRunOptions,
  InitializeConfig,
  InferenceBackend,
  ImageSource,
  TextResult,
  PreDetectionOptions,
  DetectionRunOptions,
  PostDetectionOptions,
  PreRecognitionOptions,
  RecognitionRunOptions,
  PostRecognitionOptions,
  OutputGranularity,
} from '../types/index.js';
import { extractImageData } from '../utils/imageUtils.js';
import {
  loadModelAssets,
  initWasmRuntime,
  RtenOcrPipeline,
  type LoadedModels,
} from './wasmLoader.js';
import { OcrWorkerBridge } from './workerBridge.js';

interface RgbSource {
  pixels: Uint8Array | ArrayBuffer;
  width: number;
  height: number;
  stride?: number | null;
}

/**
 * Fluent builder for chaining text detection options and running inference.
 */
export class DetectTextBuilder<TResult = TextResult[]> {
  private _engine: WebOcrEngine;
  private _source: ImageSource | null = null;
  private _rgbSource: RgbSource | null = null;
  private _options: OcrRunOptions = {};

  constructor(engine: WebOcrEngine, source: ImageSource | null = null) {
    this._engine = engine;
    this._source = source;
  }

  /** Specify the image source (file path, URL, Blob, Canvas, ImageData, or byte buffer). */
  image(source: ImageSource): this {
    this._source = source;
    this._rgbSource = null;
    return this;
  }

  /** Specify raw interleaved 24-bit RGB pixel buffer. */
  rgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride?: number | null
  ): this {
    this._rgbSource = { pixels, width, height, stride };
    this._source = null;
    return this;
  }

  /** Enable or disable document unwarping. */
  withAutoUnwarp(enabled = true): this {
    this._options.autoUnwarp = enabled;
    return this;
  }

  /** Enable or disable document layout reading order analysis. */
  withLayoutReading(enabled = true): this {
    this._options.layoutReading = enabled;
    return this;
  }

  /** Enable or disable document orientation detection. */
  withOrientationDetection(enabled = true): this {
    this._options.orientationDetection = enabled;
    return this;
  }

  /** Enable or disable text line orientation classification. */
  withLineClassification(enabled = true): this {
    this._options.lineClassification = enabled;
    return this;
  }

  /** Configure pre-detection options. */
  withPreDetection(options: PreDetectionOptions): this {
    this._options.preDetection = { ...this._options.preDetection, ...options };
    return this;
  }

  /** Configure detection inference options. */
  withDetection(options: DetectionRunOptions): this {
    this._options.detection = { ...this._options.detection, ...options };
    return this;
  }

  /** Configure post-detection filtering options. */
  withPostDetection(options: PostDetectionOptions): this {
    this._options.postDetection = { ...this._options.postDetection, ...options };
    return this;
  }

  /** Configure pre-recognition clustering options. */
  withPreRecognition(options: PreRecognitionOptions): this {
    this._options.preRecognition = { ...this._options.preRecognition, ...options };
    return this;
  }

  /** Configure recognition inference options. */
  withRecognition(options: RecognitionRunOptions): this {
    this._options.recognition = { ...this._options.recognition, ...options };
    return this;
  }

  /** Configure post-recognition filtering options. */
  withPostRecognition(options: PostRecognitionOptions): this {
    this._options.postRecognition = { ...this._options.postRecognition, ...options };
    return this;
  }

  /** Set output format to line-level results. */
  asLines(): DetectTextBuilder<TextResult[]> {
    this._options.output = 'lines';
    return this as unknown as DetectTextBuilder<TextResult[]>;
  }

  /** Set output format to word-level results. */
  asWords(): DetectTextBuilder<TextResult[]> {
    this._options.output = 'words';
    return this as unknown as DetectTextBuilder<TextResult[]>;
  }

  /** Set output format to 2D spatial text layout. */
  asSpatial(): DetectTextBuilder<string> {
    this._options.output = 'spatial';
    return this as unknown as DetectTextBuilder<string>;
  }

  /** Set output format to TSV (tab-separated values). */
  asTsv(): DetectTextBuilder<string> {
    this._options.output = 'tsv';
    return this as unknown as DetectTextBuilder<string>;
  }

  /** Set output granularity. */
  withOutput(granularity: 'spatial' | 'tsv'): DetectTextBuilder<string>;
  withOutput(granularity: 'lines' | 'words'): DetectTextBuilder<TextResult[]>;
  withOutput(granularity: OutputGranularity): DetectTextBuilder<TextResult[] | string>;
  withOutput(granularity: OutputGranularity): DetectTextBuilder<TextResult[] | string> {
    this._options.output = granularity;
    return this as unknown as DetectTextBuilder<TextResult[] | string>;
  }

  /** Merge multiple options. */
  withOptions(options: OcrRunOptions & { output: 'spatial' | 'tsv' }): DetectTextBuilder<string>;
  withOptions(options: OcrRunOptions & { output: 'lines' | 'words' }): DetectTextBuilder<TextResult[]>;
  withOptions(options: OcrRunOptions): this;
  withOptions(options: OcrRunOptions): DetectTextBuilder<any> {
    this._options = { ...this._options, ...options };
    return this;
  }

  /** Execute OCR detection with chained options. */
  async run(source?: ImageSource): Promise<TResult> {
    if (source !== undefined) {
      return this._engine.executeDetectText(source, this._options) as Promise<TResult>;
    }
    if (this._rgbSource !== null) {
      return this._engine.executeDetectTextRgb(
        this._rgbSource.pixels,
        this._rgbSource.width,
        this._rgbSource.height,
        this._rgbSource.stride,
        this._options
      ) as Promise<TResult>;
    }
    if (this._source !== null) {
      return this._engine.executeDetectText(this._source, this._options) as Promise<TResult>;
    }
    throw new Error('Image source or RGB buffer must be provided to detectText() or run()');
  }
}

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
    this.currentConfig = { ...config };

    const backend = config.backend || (config.webgpu ? 'rusto-webgpu' : 'rusto-nn');

    // 1. Initialize WebAssembly runtime for the requested pluggable engine
    await initWasmRuntime(config.wasmUrl, backend);

    // 2. Load model buffers if provided
    if (config.models) {
      this.loadedModels = await loadModelAssets(config.models);
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
    if (config.worker && typeof Worker !== 'undefined') {
      try {
        await this.workerBridge.initialize(config);
      } catch (err) {
        console.warn('Worker initialization failed; falling back to main-thread RTen engine:', err);
      }
    }

    this.initialized = true;
  }

  /**
   * Checks if the engine has been successfully initialized and is ready for inference.
   */
  public isReady(): boolean {
    return this.initialized;
  }

  /**
   * Checks if the engine has been successfully initialized.
   */
  public isInitialized(): boolean {
    return this.isReady();
  }

  /**
   * Returns the Web OCR engine library version string.
   */
  public version(): string {
    return '0.4.0';
  }

  /**
   * Returns the active inference engine backend identifier ('rusto-nn' | 'rusto-rten' | 'rusto-webgpu').
   */
  public backend(): InferenceBackend {
    return (this.currentConfig.backend ||
      (this.currentConfig.webgpu ? 'rusto-webgpu' : 'rusto-nn')) as InferenceBackend;
  }

  /**
   * Downloads or pre-fetches model asset URLs for a specific pipeline ahead of time.
   */
  public async downloadModel(model = 'indraocr', _lang?: string, _cacheDir?: string): Promise<string[]> {
    return [
      `https://huggingface.co/byrizki/rten-models/resolve/main/${model}/det.rten`,
      `https://huggingface.co/byrizki/rten-models/resolve/main/${model}/rec.rten`,
      `https://huggingface.co/byrizki/rten-models/resolve/main/${model}/meta.json`,
    ];
  }

  /**
   * Returns a fluent builder for chaining detection options.
   */
  public detectText(): DetectTextBuilder;
  /**
   * Returns a fluent builder for chaining detection options for a given source.
   */
  public detectText(source: ImageSource): DetectTextBuilder;
  /**
   * Runs text detection and recognition on an image source returning spatial layout.
   */
  public detectText(
    source: ImageSource,
    options: OcrRunOptions & { output: 'spatial' }
  ): Promise<string>;
  /**
   * Runs text detection and recognition on an image source returning lines or words.
   */
  public detectText(
    source: ImageSource,
    options?: OcrRunOptions & { output?: 'lines' | 'words' }
  ): Promise<TextResult[]>;
  /**
   * Runs text detection and recognition on an image source.
   */
  public detectText(
    source: ImageSource,
    options?: OcrRunOptions
  ): Promise<TextResult[] | string>;
  public detectText(
    source?: ImageSource,
    options?: OcrRunOptions
  ): DetectTextBuilder | Promise<TextResult[] | string> {
    if (arguments.length === 0) {
      return new DetectTextBuilder(this);
    }
    if (arguments.length === 1 || options === undefined) {
      return new DetectTextBuilder(this, source!);
    }
    return this.executeDetectText(source!, options);
  }

  /**
   * Detects text from raw interleaved 24-bit RGB pixel buffer returning spatial layout.
   */
  public detectTextRgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride: number | null | undefined,
    options: OcrRunOptions & { output: 'spatial' }
  ): Promise<string>;
  /**
   * Detects text from raw interleaved 24-bit RGB pixel buffer returning lines or words.
   */
  public detectTextRgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride?: number | null,
    options?: OcrRunOptions & { output?: 'lines' | 'words' }
  ): Promise<TextResult[]>;
  /**
   * Detects text from raw interleaved 24-bit RGB pixel buffer.
   */
  public detectTextRgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride?: number | null,
    options?: OcrRunOptions
  ): Promise<TextResult[] | string>;
  public detectTextRgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride?: number | null,
    options?: OcrRunOptions
  ): Promise<TextResult[] | string> {
    return this.executeDetectTextRgb(pixels, width, height, stride, options);
  }

  /**
   * Direct execution of text detection on an image source with given options.
   */
  public async executeDetectText(
    source: ImageSource,
    options: OcrRunOptions = {}
  ): Promise<TextResult[] | string> {
    if (!this.initialized) {
      throw new Error(
        'Rusto Web engine is not initialized. Call initialize() before calling detectText().'
      );
    }

    const rawImage = await extractImageData(source);

    if (this.workerBridge.isAvailable()) {
      return await this.workerBridge.detect(rawImage, options);
    }

    return await this.executePipeline(rawImage.data, rawImage.width, rawImage.height, rawImage.channels, null, options);
  }

  /**
   * Direct execution of text detection on raw RGB pixels with given options.
   */
  public async executeDetectTextRgb(
    pixels: Uint8Array | ArrayBuffer,
    width: number,
    height: number,
    stride?: number | null,
    options: OcrRunOptions = {}
  ): Promise<TextResult[] | string> {
    if (!this.initialized) {
      throw new Error(
        'Rusto Web engine is not initialized. Call initialize() before calling detectText().'
      );
    }

    if (this.workerBridge.isAvailable()) {
      return await this.workerBridge.detectRgb(pixels, width, height, stride, options);
    }

    return await this.executePipeline(pixels, width, height, 3, stride, options);
  }

  /**
   * Pipeline execution running RTen model inference and post-processing in WebAssembly.
   * Yields to the event loop so that animations/render frames remain fluid.
   */
  private async executePipeline(
    pixels: Uint8Array | Uint8ClampedArray | ArrayBuffer,
    width: number,
    height: number,
    channels: number,
    stride: number | null | undefined,
    options: OcrRunOptions = {}
  ): Promise<TextResult[] | string> {
    if (!this.pipeline) {
      return options.output === 'spatial' ? '' : [];
    }

    // Yield to the event loop once before computation to allow the browser to paint
    await new Promise<void>((resolve) => {
      if (typeof setTimeout === 'function') {
        setTimeout(resolve, 0);
      } else {
        resolve();
      }
    });

    const pixelBytes =
      pixels instanceof Uint8Array
        ? pixels
        : pixels instanceof Uint8ClampedArray
          ? new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength)
          : new Uint8Array(pixels);

    const optionsJson = options ? JSON.stringify(options) : null;
    const jsonStr = this.pipeline.detectText(
      pixelBytes,
      width,
      height,
      channels,
      stride ?? null,
      optionsJson
    );

    try {
      const parsed = JSON.parse(jsonStr);
      return parsed as TextResult[] | string;
    } catch {
      return jsonStr;
    }
  }

  /**
   * Closes the engine and cleans up worker and model resources.
   */
  public close(): void {
    this.workerBridge.terminate();
    if (this.pipeline) {
      this.pipeline.free();
      this.pipeline = null;
    }
    this.initialized = false;
    this.loadedModels = {};
  }
}
