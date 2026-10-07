import {
  calculateFrameFromQuad,
  formatSpatialText,
  groupCandidatesIntoLines,
} from '../src/utils/geometry.js';
import type { TextResult } from '../src/types/index.js';

describe('geometry tests', () => {
  test('calculateFrameFromQuad returns bounding box bounds', () => {
    const quad: [[number, number], [number, number], [number, number], [number, number]] = [
      [10, 20],
      [110, 25],
      [105, 60],
      [15, 55],
    ];
    const frame = calculateFrameFromQuad(quad);
    expect(frame.left).toBe(10);
    expect(frame.top).toBe(20);
    expect(frame.width).toBe(100);
    expect(frame.height).toBe(40);
  });

  test('groupCandidatesIntoLines groups boxes on the same line', () => {
    const item1: TextResult = {
      text: 'Hello',
      score: 0.95,
      box_points: [[10, 10], [50, 10], [50, 30], [10, 30]],
      frame: { left: 10, top: 10, width: 40, height: 20 },
    };
    const item2: TextResult = {
      text: 'World',
      score: 0.92,
      box_points: [[60, 12], [110, 12], [110, 32], [60, 32]],
      frame: { left: 60, top: 12, width: 50, height: 20 },
    };
    const item3: TextResult = {
      text: 'Second Line',
      score: 0.90,
      box_points: [[10, 60], [120, 60], [120, 80], [10, 80]],
      frame: { left: 10, top: 60, width: 110, height: 20 },
    };

    const grouped = groupCandidatesIntoLines([item1, item3, item2]);
    expect(grouped.length).toBe(2);
    expect(grouped[0].map((c) => c.text)).toEqual(['Hello', 'World']);
    expect(grouped[1].map((c) => c.text)).toEqual(['Second Line']);
  });

  test('formatSpatialText formats rows with spaces and newlines', () => {
    const item1: TextResult = {
      text: 'Total:',
      score: 0.99,
      box_points: [[10, 10], [60, 10], [60, 30], [10, 30]],
      frame: { left: 10, top: 10, width: 50, height: 20 },
    };
    const item2: TextResult = {
      text: '$50.00',
      score: 0.98,
      box_points: [[120, 10], [180, 10], [180, 30], [120, 30]],
      frame: { left: 120, top: 10, width: 60, height: 20 },
    };

    const lines = [[item1, item2]];
    const output = formatSpatialText(lines);
    expect(output).toContain('Total:');
    expect(output).toContain('$50.00');
  });
});
