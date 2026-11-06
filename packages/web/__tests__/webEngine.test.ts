import { WebOcrEngine } from '../src/engine/webEngine.js';
import { initialize, detectText, detectTextRgb, isReady, isInitialized, close, version, downloadModel } from '../src/index.js';

describe('WebOcrEngine & singleton API', () => {
  afterEach(() => {
    close();
  });

  test('throws if detectText().run() is called before initialize', async () => {
    const engine = new WebOcrEngine();
    await expect(engine.detectText({ bytes: new Uint8Array([1, 2, 3]) }).run()).rejects.toThrow(
      'Rusto Web engine is not initialized.'
    );
  });

  test('initializes and runs detectText with default options', async () => {
    const engine = new WebOcrEngine();
    await engine.initialize({ preset: 'ppv6' });
    expect(engine.isInitialized()).toBe(true);

    const result = await engine.detectText({ bytes: new Uint8Array([1, 2, 3]) }).run();
    expect(Array.isArray(result)).toBe(true);
  });

  test('detectText with spatial output returns a string', async () => {
    const engine = new WebOcrEngine();
    await engine.initialize({ preset: 'ppv6' });

    const result = await engine.detectText(
      { bytes: new Uint8Array([1, 2, 3]) },
      { output: 'spatial' }
    );
    expect(typeof result).toBe('string');
  });

  test('singleton functions initialize, detectText, isReady, version, downloadModel, and close work correctly', async () => {
    expect(isReady()).toBe(false);
    expect(isInitialized()).toBe(false);
    expect(version()).toBe('0.4.0');

    const modelUrls = await downloadModel('indraocr');
    expect(Array.isArray(modelUrls)).toBe(true);
    expect(modelUrls.length).toBeGreaterThan(0);

    await initialize({ preset: 'ppv6' });
    expect(isReady()).toBe(true);
    expect(isInitialized()).toBe(true);

    const results = await detectText({ bytes: new Uint8Array([1, 2, 3]) }).run();
    expect(Array.isArray(results)).toBe(true);

    close();
    expect(isReady()).toBe(false);
    expect(isInitialized()).toBe(false);
  });

  test('detectText fluent builder chaining works', async () => {
    const engine = new WebOcrEngine();
    await engine.initialize({ preset: 'ppv6' });

    const builder = engine.detectText()
      .image({ bytes: new Uint8Array([1, 2, 3]) })
      .withAutoUnwarp(true)
      .withLayoutReading(true)
      .withOrientationDetection(true)
      .withLineClassification(true)
      .withPreDetection({ descreenStrength: 0.1 })
      .withDetection({ limitSideLen: 960 })
      .withPostDetection({ threshold: 0.3 })
      .withPreRecognition({ lineYThreshold: 0.5 })
      .withRecognition({ batchSize: 4 })
      .withPostRecognition({ scoreThreshold: 0.6 })
      .asLines();

    const res = await builder.run();
    expect(Array.isArray(res)).toBe(true);

    const spatialRes = await engine.detectText({ bytes: new Uint8Array([1, 2, 3]) })
      .asSpatial()
      .run();
    expect(typeof spatialRes).toBe('string');
  });

  test('detectTextRgb and builder.rgb chaining works', async () => {
    const engine = new WebOcrEngine();
    await engine.initialize({ preset: 'ppv6' });

    const rgbPixels = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
    const res = await engine.detectTextRgb(rgbPixels, 2, 2, 6);
    expect(Array.isArray(res)).toBe(true);

    const builderRes = await engine.detectText()
      .rgb(rgbPixels, 2, 2, 6)
      .asLines()
      .run();
    expect(Array.isArray(builderRes)).toBe(true);

    const spatialRes = await detectTextRgb(rgbPixels, 2, 2, 6, { output: 'spatial' });
    expect(typeof spatialRes).toBe('string');
  });
});
