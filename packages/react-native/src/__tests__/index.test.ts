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
});
