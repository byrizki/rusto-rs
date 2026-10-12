import {
  calculateFrameFromQuad,
  clusterLineSegments,
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

  test('clusterLineSegments merges adjacent words but keeps distant columns separated', () => {
    const col1: TextResult = {
      text: 'Kewarganegaraan WNI',
      score: 0.95,
      box_points: [[30, 100], [200, 100], [200, 130], [30, 130]],
      frame: { left: 30, top: 100, width: 170, height: 30 },
    };
    const col2: TextResult = {
      text: 'JAKARTA PUSAT',
      score: 0.98,
      box_points: [[600, 102], [750, 102], [750, 132], [600, 132]],
      frame: { left: 600, top: 102, width: 150, height: 30 },
    };

    // Huge gap of 400px between col1 and col2: should NOT be merged
    const unmerged = clusterLineSegments([col1, col2], 0.4);
    expect(unmerged.length).toBe(2);
    expect(unmerged[0].text).toBe('Kewarganegaraan WNI');
    expect(unmerged[1].text).toBe('JAKARTA PUSAT');

    // Small gap between adjacent words: should be merged
    const word1: TextResult = {
      text: 'Bertaku',
      score: 0.94,
      box_points: [[10, 200], [60, 200], [60, 230], [10, 230]],
      frame: { left: 10, top: 200, width: 50, height: 30 },
    };
    const word2: TextResult = {
      text: 'Hingga',
      score: 0.96,
      box_points: [[68, 200], [120, 200], [120, 230], [68, 230]],
      frame: { left: 68, top: 200, width: 52, height: 30 },
    };
    const merged = clusterLineSegments([word1, word2], 0.4);
    expect(merged.length).toBe(1);
    expect(merged[0].text).toBe('Bertaku Hingga');
    expect(merged[0].frame.left).toBe(10);
    expect(merged[0].frame.width).toBe(110);
  });
});
