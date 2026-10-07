import {
  normalizeInitializeConfig,
  normalizeOptions,
  normalizeDetectionOptions,
  normalizePostprocessOptions,
  requireOnlyKeys,
  isObject,
} from '../src/utils/validator.js';

describe('validator tests', () => {
  test('isObject identifies objects and rejects primitives/arrays', () => {
    expect(isObject({})).toBe(true);
    expect(isObject({ a: 1 })).toBe(true);
    expect(isObject(null)).toBe(false);
    expect(isObject(undefined)).toBe(false);
    expect(isObject([])).toBe(false);
    expect(isObject('str')).toBe(false);
    expect(isObject(123)).toBe(false);
  });

  test('requireOnlyKeys throws on unrecognized keys', () => {
    expect(() => requireOnlyKeys({ a: 1, b: 2 }, ['a'], 'Unknown key')).toThrow('Unknown key');
    expect(() => requireOnlyKeys({ a: 1 }, ['a', 'b'], 'Unknown key')).not.toThrow();
  });

  test('normalizeInitializeConfig accepts valid configs and rejects invalid presets', () => {
    expect(normalizeInitializeConfig(undefined)).toEqual({});
    expect(normalizeInitializeConfig({ preset: 'ppv6' })).toEqual({ preset: 'ppv6' });
    expect(() => normalizeInitializeConfig({ preset: 'invalid' as any })).toThrow(
      'InitializeConfig.preset is invalid.'
    );
  });

  test('normalizeOptions validates threshold, limits, and output modes', () => {
    expect(normalizeOptions(undefined)).toEqual({});
    expect(normalizeOptions({ output: 'lines', textScore: 0.85 })).toEqual({
      output: 'lines',
      textScore: 0.85,
    });

    expect(() => normalizeOptions({ output: 'unknown' as any })).toThrow(
      'DetectTextOptions.output is invalid.'
    );
    expect(() => normalizeOptions({ textScore: -0.1 })).toThrow(
      'DetectTextOptions.textScore must be a finite number from 0 to 1.'
    );
    expect(() => normalizeOptions({ textScore: 1.5 })).toThrow(
      'DetectTextOptions.textScore must be a finite number from 0 to 1.'
    );
    expect(() => normalizeOptions({ minSideLen: 100, maxSideLen: 50 })).toThrow(
      'DetectTextOptions.minSideLen must be <= maxSideLen.'
    );
  });

  test('normalizeDetectionOptions validates limitSideLen and limitType', () => {
    expect(() => normalizeDetectionOptions({ limitSideLen: 0 })).toThrow();
    expect(() => normalizeDetectionOptions({ limitSideLen: 32768 })).toThrow();
    expect(() => normalizeDetectionOptions({ limitType: 'bad' as any })).toThrow();
    expect(() => normalizeDetectionOptions({ limitSideLen: 960, limitType: 'max' })).not.toThrow();
  });

  test('normalizePostprocessOptions validates postprocess bounds', () => {
    expect(() => normalizePostprocessOptions({ threshold: 1.2 })).toThrow();
    expect(() => normalizePostprocessOptions({ maxCandidates: 0 })).toThrow();
    expect(() => normalizePostprocessOptions({ unclipRatio: -1 })).toThrow();
    expect(() =>
      normalizePostprocessOptions({
        threshold: 0.3,
        boxThreshold: 0.6,
        maxCandidates: 1000,
        unclipRatio: 1.5,
        useDilation: true,
      })
    ).not.toThrow();
  });

  test('normalizeOptions accepts calibration and optimization options', () => {
    expect(() =>
      normalizeOptions({
        calibration: { enabled: true, descreenBackground: true, descreenStrength: 1.15 },
        optimization: { enabled: true },
      })
    ).not.toThrow();
  });
});

