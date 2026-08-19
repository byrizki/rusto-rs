import { NativeModules } from 'react-native';

jest.mock('react-native', () => ({
  NativeModules: { Rusto: { initialize: jest.fn(), detectText: jest.fn() } },
  Platform: { select: jest.fn() },
}));

import { detectText, initialize } from '../index';

describe('React Native API', () => {
  beforeEach(() => jest.clearAllMocks());

  it('forwards shallow initialization config', async () => {
    (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(undefined);
    await initialize({ preset: 'ppv6', models: { recognition: 'rec.mnn' } });
    expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({ preset: 'ppv6', models: { recognition: 'rec.mnn' } });
  });

  it('forwards line source and options', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
    await detectText({ uri: '/tmp/image.png' }, { output: 'lines', lineYThreshold: 0.5 });
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith({ uri: '/tmp/image.png' }, { output: 'lines', lineYThreshold: 0.5 });
  });

  it('forwards spatial request', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue('spatial text');
    await expect(detectText({ base64: 'aGVsbG8=' }, { output: 'spatial' })).resolves.toBe('spatial text');
  });

  it('normalizes surrounding whitespace in string sources', async () => {
    (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
    await detectText({ uri: ' /tmp/image.png ' });
    expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith({ uri: '/tmp/image.png' }, {});
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
    expect(() => detectText(source as never)).toThrow(/ImageSource/);
    expect(NativeModules.Rusto.detectText).not.toHaveBeenCalled();
  });

  it.each([{ bytes: null }, { bytes: {} }, { bytes: 'not-bytes' }, { bytes: new Uint8Array() }])(
    'rejects invalid byte source %# before native invocation',
    (source) => {
      expect(() => detectText(source as never)).toThrow(/ImageSource\.bytes/);
      expect(NativeModules.Rusto.detectText).not.toHaveBeenCalled();
    }
  );

  it.each([
    null,
    { output: null },
    { output: 'columns' },
    { lineYThreshold: -1 },
    { wordXThreshold: Number.NaN },
    { textScore: 2 },
    { classification: 'true' },
    { unknown: true },
  ])('rejects invalid runtime options %# before native invocation', (options) => {
    expect(() => detectText({ uri: '/tmp/image.png' }, options as never)).toThrow(/DetectTextOptions/);
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
    { unknown: true },
  ])('rejects invalid initialization config %# before native invocation', (config) => {
    expect(() => initialize(config as never)).toThrow(/InitializeConfig/);
    expect(NativeModules.Rusto.initialize).not.toHaveBeenCalled();
  });
});
