import {
  WebOcrEngine,
  type DetectTextOptions,
  type ImageSource,
  type InitializeConfig,
  type TextResult,
} from '@rustors/web';

export class OcrService {
  private engine: WebOcrEngine;

  constructor(engineInstance?: WebOcrEngine) {
    this.engine = engineInstance || new WebOcrEngine();
  }

  public async init(config?: InitializeConfig): Promise<void> {
    await this.engine.initialize(config);
  }

  public isReady(): boolean {
    return this.engine.isReady();
  }

  public async detect(
    source: ImageSource,
    options?: DetectTextOptions
  ): Promise<TextResult[] | string> {
    if (!this.engine.isReady()) {
      throw new Error('OCR Service engine is not ready. Call init() first.');
    }
    return this.engine.executeDetectText(source, options);
  }

  public close(): void {
    this.engine.close();
  }

  public getEngine(): WebOcrEngine {
    return this.engine;
  }
}

export const defaultOcrService = new OcrService();
