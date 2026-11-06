import { NativeModules } from 'react-native';

jest.mock('react-native', () => ({
  NativeModules: {
    Rusto: {
      initialize: jest.fn(),
      detectText: jest.fn(),
      close: jest.fn(),
      isReady: jest.fn(),
      version: jest.fn(),
    },
  },
  Platform: { select: jest.fn() },
}));

import { detectText, initialize, close, isReady, version, DetectTextBuilder } from '../index';


describe('React Native API', () => {
  beforeEach(() => jest.clearAllMocks());

  it('forwards shallow initialization config including indra preset', async () => {
    (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(undefined);
    await initialize({
      preset: 'indra',
      models: { recognition: 'rec.rten' },
    });
    expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({
      preset: 'indra',
      models: { recognition: 'rec.rten' },
    });
  });

  it('forwards root runtime options across phases', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
    const options = {
      output: 'lines' as const,
      autoUnwarp: true,
      layoutReading: true,
      orientationDetection: true,
      lineClassification: true,
      preDetection: { targetMaxSide: 960 },
      detection: { limitType: 'min' as const },
      postDetection: { useDilation: false },
      preRecognition: { lineYThreshold: 0.5 },
      recognition: { batchSize: 8 },
      postRecognition: { scoreThreshold: 0.3 },
    };
    await detectText({ uri: '/tmp/image.png' }, options);
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith({ uri: '/tmp/image.png' }, options);
  });

  it('forwards spatial request', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue('spatial text');
    await expect(detectText({ base64: 'aGVsbG8=' }, { output: 'spatial' })).resolves.toBe(
      'spatial text'
    );
  });

  it('normalizes surrounding whitespace in string sources when run() is called', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
    await detectText({ uri: ' /tmp/image.png ' }).run();
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith({ uri: '/tmp/image.png' }, {});
  });

  it('forwards optional pipeline stages in initialize and detect options', async () => {
    (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(null);
    await initialize({
      models: {
        unwarp: 'uvdoc.rten',
        layout: 'layout.rten',
        classification: 'cls.rten',
        orientation: 'orient.rten',
      },
    });
    expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({
      models: {
        unwarp: 'uvdoc.rten',
        layout: 'layout.rten',
        classification: 'cls.rten',
        orientation: 'orient.rten',
      },
    });

    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
    await detectText({ uri: '/tmp/test.png' }, { autoUnwarp: true, layoutReading: true });
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith(
      { uri: '/tmp/test.png' },
      { autoUnwarp: true, layoutReading: true }
    );
  });

  it('forwards model, modelPath, lang, and layoutVariant in initialize', async () => {
    (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(null);
    const config = {
      model: 'indraocr',
      modelPath: 'models/PPOCR_v6_tiny',
      lang: 'ja',
      cacheDir: '/tmp/cache',
      enableReadingOrder: true,
      layoutVariant: 'publaynet' as const,
    };
    await initialize(config);
    expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith(config);
  });

  it('supports fluent builder chaining with detectText().with...().asLines().run()', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);

    const builder = detectText()
      .image({ uri: '/tmp/image.png' })
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

    expect(builder).toBeInstanceOf(DetectTextBuilder);
    const res = await builder.run();
    expect(res).toEqual([]);
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith(
      { uri: '/tmp/image.png' },
      {
        autoUnwarp: true,
        layoutReading: true,
        orientationDetection: true,
        lineClassification: true,
        preDetection: { descreenStrength: 0.1 },
        detection: { limitSideLen: 960 },
        postDetection: { threshold: 0.3 },
        preRecognition: { lineYThreshold: 0.5 },
        recognition: { batchSize: 4 },
        postRecognition: { scoreThreshold: 0.6 },
        output: 'lines',
      }
    );

    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue('spatial output');
    const spatial = await detectText({ uri: '/tmp/image.png' }).asSpatial().run();
    expect(spatial).toBe('spatial output');
  });

  it('calls close() correctly', async () => {
    (NativeModules.Rusto.close as jest.Mock).mockResolvedValue(undefined);
    await close();
    expect(NativeModules.Rusto.close).toHaveBeenCalled();
  });

  it.each([
    null,
    undefined,
    'image.png',
    [],
    {},
    { uri: '/tmp/image.png', base64: 'aGVsbG8=' },
    { uri: '/tmp/image.png', extra: true },
    { uri: '' },
  ])('rejects invalid image source %# before native invocation', (source) => {
    expect(() => detectText(source as never)).toThrow(/ImageSource/u);
    expect(NativeModules.Rusto.detectText).not.toHaveBeenCalled();
  });

  it.each([{ bytes: null }, { bytes: {} }, { bytes: 'not-bytes' }, { bytes: new Uint8Array() }])(
    'rejects invalid byte source %# before native invocation',
    (source) => {
      expect(() => detectText(source as never)).toThrow(/ImageSource\.bytes/u);
      expect(NativeModules.Rusto.detectText).not.toHaveBeenCalled();
    }
  );

  it.each([
    null,
    { output: null },
    { output: 'columns' },
    { autoUnwarp: 'true' },
    { detection: { limitType: 'other' } },
    { detection: { mean: [0.5, 0.5] } },
    { detection: { std: [0.5, 0, 0.5] } },
    { postDetection: { useDilation: 'true' } },
    { postDetection: { scoreMode: 'invalid' } },
    { preRecognition: { lineYThreshold: -1 } },
    { recognition: { batchSize: 0 } },
    { postRecognition: { scoreThreshold: 2 } },
    { unknown: true },
  ])('rejects invalid runtime options %# before native invocation', (options) => {
    expect(() => detectText({ uri: '/tmp/image.png' }, options as never)).toThrow(
      /OcrRunOptions/u
    );
    expect(NativeModules.Rusto.detectText).not.toHaveBeenCalled();
  });

  it.each([
    null,
    'ppv6',
    { preset: null },
    { models: null },
    { preset: 'unknown' },
    { models: 'invalid' },
    { models: { unknown: 'model.mnn' } },
    { models: { recognition: '' } },
    { preprocessing: null },
    { unknown: true },
  ])('rejects invalid initialization config %# before native invocation', (config) => {
    expect(() => initialize(config as never)).toThrow(/InitializeConfig/u);
    expect(NativeModules.Rusto.initialize).not.toHaveBeenCalled();
  });

  it('exposes version and isReady inquiries', async () => {
    (NativeModules.Rusto.version as jest.Mock).mockResolvedValue('0.4.0');
    (NativeModules.Rusto.isReady as jest.Mock).mockResolvedValue(true);
    await expect(version()).resolves.toBe('0.4.0');
    await expect(isReady()).resolves.toBe(true);
  });
});
