import { OcrService } from '../src/services/ocrService.js';
import { WebOcrEngine } from '@rusto/web';

describe('OcrService tests', () => {
  test('isReady returns false before init, true after init', async () => {
    const service = new OcrService();
    expect(service.isReady()).toBe(false);

    await service.init({ preset: 'ppv6' });
    expect(service.isReady()).toBe(true);

    service.terminate();
    expect(service.isReady()).toBe(false);
  });

  test('throws error if detect is called when not ready', async () => {
    const service = new OcrService();
    await expect(service.detect({ bytes: new Uint8Array([1, 2, 3]) })).rejects.toThrow(
      'OCR Service engine is not ready.'
    );
  });

  test('calls underlying engine detectText when ready', async () => {
    const mockEngine = new WebOcrEngine();
    const service = new OcrService(mockEngine);

    await service.init();
    const result = await service.detect({ bytes: new Uint8Array([1, 2, 3]) });
    expect(Array.isArray(result)).toBe(true);
  });
});
