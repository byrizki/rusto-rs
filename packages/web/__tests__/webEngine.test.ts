import { WebOcrEngine } from '../src/engine/webEngine.js';
import { initialize, detectText, isInitialized, terminate } from '../src/index.js';

describe('WebOcrEngine & singleton API', () => {
  afterEach(() => {
    terminate();
  });

  test('throws if detectText is called before initialize', async () => {
    const engine = new WebOcrEngine();
    await expect(engine.detectText({ bytes: new Uint8Array([1, 2, 3]) })).rejects.toThrow(
      'Rusto Web engine is not initialized.'
    );
  });

  test('initializes and runs detectText with default options', async () => {
    const engine = new WebOcrEngine();
    await engine.initialize({ preset: 'ppv6' });
    expect(engine.isInitialized()).toBe(true);

    const result = await engine.detectText({ bytes: new Uint8Array([1, 2, 3]) });
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

  test('singleton functions initialize, detectText, isInitialized, and terminate work correctly', async () => {
    expect(isInitialized()).toBe(false);
    await initialize({ preset: 'ppv6' });
    expect(isInitialized()).toBe(true);

    const results = await detectText({ bytes: new Uint8Array([1, 2, 3]) });
    expect(Array.isArray(results)).toBe(true);

    terminate();
    expect(isInitialized()).toBe(false);
  });
});
