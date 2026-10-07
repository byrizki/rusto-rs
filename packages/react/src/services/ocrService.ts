import {
  WebOcrEngine,
  type DetectTextOptions,
  type ImageSource,
  type InitializeConfig,
  type TextResult,
} from 'rusto-web';

export class OcrService {
  private engine: WebOcrEngine;

  constructor(engineInstance?: WebOcrEngine) {
    this.engine = engineInstance || new WebOcrEngine();
  }

  public async init(config?: InitializeConfig): Promise<void> {
    await this.engine.initialize(config);
  }

  public isReady(): boolean {
    return this.engine.isInitialized();
  }

  public async detect(
    source: ImageSource,
    options?: DetectTextOptions
  ): Promise<TextResult[] | string> {
    if (!this.engine.isInitialized()) {
      throw new Error('OCR Service engine is not ready. Call init() first.');
    }
    return this.engine.detectText(source, options);
  }

  public terminate(): void {
    this.engine.terminate();
  }

  public getEngine(): WebOcrEngine {
    return this.engine;
  }
}

export const defaultOcrService = new OcrService();
