import {
  computeLineFeatures,
  decodeOrderFromLogits,
} from '../src/utils/readingOrder.js';
import type { TextResult } from '../src/types/index.js';

describe('readingOrder tests', () => {
  test('computeLineFeatures computes 16 features per candidate', () => {
    const items: TextResult[] = [
      {
        text: 'Title Heading',
        score: 0.98,
        box_points: [
          [20, 20],
          [200, 20],
          [200, 50],
          [20, 50],
        ],
        frame: { left: 20, top: 20, width: 180, height: 30 },
      },
      {
        text: '123 Item description',
        score: 0.95,
        box_points: [
          [20, 60],
          [150, 60],
          [150, 80],
          [20, 80],
        ],
        frame: { left: 20, top: 60, width: 130, height: 20 },
      },
    ];

    const feats = computeLineFeatures(items, 800, 1000);
    expect(feats.length).toBe(32);
    // x0 / W
    expect(feats[0]).toBeCloseTo(20 / 800);
    // y0 / H
    expect(feats[1]).toBeCloseTo(20 / 1000);
    // First item starts with uppercase
    expect(feats[15]).toBe(1.0);
  });

  test('decodeOrderFromLogits reorders pairwise matrix correctly', () => {
    // 3 items where item 0 precedes item 2 precedes item 1
    const n = 3;
    const a = new Float32Array(n * n);
    // 0 before 1 (+5)
    a[0 * n + 1] = 5.0;
    a[1 * n + 0] = -5.0;
    // 0 before 2 (+5)
    a[0 * n + 2] = 5.0;
    a[2 * n + 0] = -5.0;
    // 2 before 1 (+3)
    a[2 * n + 1] = 3.0;
    a[1 * n + 2] = -3.0;

    const order = decodeOrderFromLogits(a, n);
    expect(order).toEqual([0, 2, 1]);
  });
});
