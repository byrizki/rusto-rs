import { calculateScale, scaleFrame, scaleQuad } from '../src/utils/geometry.js';
import { resultsToTsv, resultsToJson, resultsToText } from '../src/utils/formatters.js';
import type { TextResult } from '@rustors/web';

describe('@rustors/react utils', () => {
  test('calculateScale computes correct scaling factors', () => {
    const scale = calculateScale(1000, 500, 500, 250);
    expect(scale.scaleX).toBe(0.5);
    expect(scale.scaleY).toBe(0.5);

    const zeroScale = calculateScale(0, 0, 100, 100);
    expect(zeroScale.scaleX).toBe(1);
    expect(zeroScale.scaleY).toBe(1);
  });

  test('scaleFrame scales frame correctly', () => {
    const frame = { left: 10, top: 20, width: 30, height: 40 };
    const scaled = scaleFrame(frame, 2, 0.5);
    expect(scaled.left).toBe(20);
    expect(scaled.top).toBe(10);
    expect(scaled.width).toBe(60);
    expect(scaled.height).toBe(20);
  });

  test('scaleQuad scales quad points correctly', () => {
    const quad: [[number, number], [number, number], [number, number], [number, number]] = [
      [10, 10],
      [20, 10],
      [20, 30],
      [10, 30],
    ];
    const scaled = scaleQuad(quad, 2, 2);
    expect(scaled[0]).toEqual([20, 20]);
    expect(scaled[1]).toEqual([40, 20]);
    expect(scaled[2]).toEqual([40, 60]);
    expect(scaled[3]).toEqual([20, 60]);
  });

  test('formatters format TextResult list into text, csv, and json', () => {
    const mockResults: TextResult[] = [
      {
        text: 'Invoice #123',
        score: 0.9921,
        box_points: [[0, 0], [10, 0], [10, 10], [0, 10]],
        frame: { left: 0, top: 0, width: 10, height: 10 },
      },
      {
        text: 'Total: $42.00',
        score: 0.9854,
        box_points: [[0, 20], [20, 20], [20, 30], [0, 30]],
        frame: { left: 0, top: 20, width: 20, height: 10 },
      },
    ];

    const text = resultsToText(mockResults);
    expect(text).toBe('Invoice #123\nTotal: $42.00');

    const tsv = resultsToTsv(mockResults);
    expect(tsv).toContain('Invoice #123\t0.9921\t0\t0\t10\t10');
    expect(tsv).toContain('Total: $42.00\t0.9854\t0\t20\t20\t10');

    const json = resultsToJson(mockResults);
    const parsed = JSON.parse(json);
    expect(parsed.length).toBe(2);
    expect(parsed[0].text).toBe('Invoice #123');
  });
});
