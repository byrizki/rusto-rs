import { NativeModules } from 'react-native';

jest.mock('react-native', () => {
  return {
    NativeModules: {
      Rusto: {
        initialize: jest.fn(),
        detectText: jest.fn(),
        detectTextFromBytes: jest.fn(),
        detectTextToRaw: jest.fn(),
        detectTextToCsv: jest.fn(),
        detectTextToTextWithPosition: jest.fn(),
        detectTextToSpatialText: jest.fn(),
        detectTextFromBytesToRaw: jest.fn(),
        detectTextFromBytesToCsv: jest.fn(),
        detectTextFromBytesToTextWithPosition: jest.fn(),
        detectTextFromBytesToSpatialText: jest.fn(),
        getVersion: jest.fn(),
      },
    },
    Platform: {
      select: jest.fn((obj) => obj.default || obj.ios || obj.android),
    },
  };
});

import {
  initialize,
  detectText,
  detectTextFromBytes,
  detectTextToRaw,
  detectTextToCsv,
  detectTextToTextWithPosition,
  detectTextToSpatialText,
  detectTextFromBytesToRaw,
  detectTextFromBytesToCsv,
  detectTextFromBytesToTextWithPosition,
  detectTextFromBytesToSpatialText,
  getVersion,
} from '../index';

describe('react-native-rusto', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initialize', () => {
    it('passes empty object when called with no arguments', async () => {
      (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(true);

      const result = await initialize();

      expect(result).toBe(true);
      expect(NativeModules.Rusto.initialize).toHaveBeenCalledTimes(1);
      expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({});
    });

    it('passes partial config options object', async () => {
      (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(true);

      const result = await initialize({ layout: { xThresholdMultiplier: 0.5 } });

      expect(result).toBe(true);
      expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({
        layout: { xThresholdMultiplier: 0.5 },
      });
    });

    it('passes template with custom overrides', async () => {
      (NativeModules.Rusto.initialize as jest.Mock).mockResolvedValue(true);

      const result = await initialize({
        template: 'ppv5',
        recognition: { scoreThresh: 0.8 },
        detection: { modelPath: 'custom_det.mnn' },
      });

      expect(result).toBe(true);
      expect(NativeModules.Rusto.initialize).toHaveBeenCalledWith({
        template: 'ppv5',
        recognition: { scoreThresh: 0.8 },
        detection: { modelPath: 'custom_det.mnn' },
      });
    });
  });

  describe('spatial text methods with default parameters', () => {
    it('passes default thresholds to detectTextToSpatialText', async () => {
      (NativeModules.Rusto.detectTextToSpatialText as jest.Mock).mockResolvedValue('ocr text');

      const result = await detectTextToSpatialText('/path/to/img.png');

      expect(result).toBe('ocr text');
      expect(NativeModules.Rusto.detectTextToSpatialText).toHaveBeenCalledWith(
        '/path/to/img.png',
        0.5,
        0.4
      );
    });

    it('passes custom thresholds to detectTextToSpatialText', async () => {
      (NativeModules.Rusto.detectTextToSpatialText as jest.Mock).mockResolvedValue('ocr text');

      const result = await detectTextToSpatialText('/path/to/img.png', 0.8, 0.3);

      expect(result).toBe('ocr text');
      expect(NativeModules.Rusto.detectTextToSpatialText).toHaveBeenCalledWith(
        '/path/to/img.png',
        0.8,
        0.3
      );
    });

    it('passes default thresholds to detectTextFromBytesToSpatialText', async () => {
      (NativeModules.Rusto.detectTextFromBytesToSpatialText as jest.Mock).mockResolvedValue('ocr bytes');

      const result = await detectTextFromBytesToSpatialText('base64data');

      expect(result).toBe('ocr bytes');
      expect(NativeModules.Rusto.detectTextFromBytesToSpatialText).toHaveBeenCalledWith(
        'base64data',
        0.5,
        0.4
      );
    });
  });

  describe('other methods', () => {
    it('forwards detectText call', async () => {
      (NativeModules.Rusto.detectText as jest.Mock).mockResolvedValue([]);
      await detectText('/path/img.png');
      expect(NativeModules.Rusto.detectText).toHaveBeenCalledWith('/path/img.png');
    });

    it('forwards detectTextFromBytes call', async () => {
      (NativeModules.Rusto.detectTextFromBytes as jest.Mock).mockResolvedValue([]);
      await detectTextFromBytes('bytes');
      expect(NativeModules.Rusto.detectTextFromBytes).toHaveBeenCalledWith('bytes');
    });

    it('forwards detectTextToRaw call', async () => {
      (NativeModules.Rusto.detectTextToRaw as jest.Mock).mockResolvedValue('raw');
      await detectTextToRaw('/path/img.png');
      expect(NativeModules.Rusto.detectTextToRaw).toHaveBeenCalledWith('/path/img.png');
    });

    it('forwards detectTextToCsv call', async () => {
      (NativeModules.Rusto.detectTextToCsv as jest.Mock).mockResolvedValue('csv');
      await detectTextToCsv('/path/img.png');
      expect(NativeModules.Rusto.detectTextToCsv).toHaveBeenCalledWith('/path/img.png');
    });

    it('forwards detectTextToTextWithPosition call', async () => {
      (NativeModules.Rusto.detectTextToTextWithPosition as jest.Mock).mockResolvedValue('pos');
      await detectTextToTextWithPosition('/path/img.png');
      expect(NativeModules.Rusto.detectTextToTextWithPosition).toHaveBeenCalledWith('/path/img.png');
    });

    it('forwards detectTextFromBytesToRaw call', async () => {
      (NativeModules.Rusto.detectTextFromBytesToRaw as jest.Mock).mockResolvedValue('raw');
      await detectTextFromBytesToRaw('bytes');
      expect(NativeModules.Rusto.detectTextFromBytesToRaw).toHaveBeenCalledWith('bytes');
    });

    it('forwards detectTextFromBytesToCsv call', async () => {
      (NativeModules.Rusto.detectTextFromBytesToCsv as jest.Mock).mockResolvedValue('csv');
      await detectTextFromBytesToCsv('bytes');
      expect(NativeModules.Rusto.detectTextFromBytesToCsv).toHaveBeenCalledWith('bytes');
    });

    it('forwards detectTextFromBytesToTextWithPosition call', async () => {
      (NativeModules.Rusto.detectTextFromBytesToTextWithPosition as jest.Mock).mockResolvedValue('pos');
      await detectTextFromBytesToTextWithPosition('bytes');
      expect(NativeModules.Rusto.detectTextFromBytesToTextWithPosition).toHaveBeenCalledWith('bytes');
    });

    it('forwards getVersion call', async () => {
      (NativeModules.Rusto.getVersion as jest.Mock).mockResolvedValue('0.2.0');
      const version = await getVersion();
      expect(version).toBe('0.2.0');
      expect(NativeModules.Rusto.getVersion).toHaveBeenCalled();
    });
  });
});
