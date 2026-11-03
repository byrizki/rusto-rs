import type { TextResult } from '../types/results.js';
import type { RtenModel } from '../wasm/rusto_rten_wasm.js';

const SENTENCE_END = new Set(['.', '?', '!', ':', ';']);
const BULLET_START = new Set(['•', '-', '–', '▪', '◦', '*', '·', '→']);

function fillItemGeometry(
  feats: Float32Array,
  base: number,
  item: TextResult,
  pageWidth: number,
  pageHeight: number,
  medH: number
): void {
  const x0 = item.frame.left;
  const y0 = item.frame.top;
  const x1 = item.frame.left + item.frame.width;
  const y1 = item.frame.top + item.frame.height;
  const w = Math.max(1e-3, item.frame.width);
  const h = Math.max(1e-3, item.frame.height);

  feats[base + 0] = x0 / pageWidth;
  feats[base + 1] = y0 / pageHeight;
  feats[base + 2] = x1 / pageWidth;
  feats[base + 3] = y1 / pageHeight;
  feats[base + 4] = (x0 + x1) / (2 * pageWidth);
  feats[base + 5] = (y0 + y1) / (2 * pageHeight);
  feats[base + 6] = w / pageWidth;
  feats[base + 7] = Math.log(h / medH);
  feats[base + 8] = Math.log(w / h) / 4.0;
  feats[base + 9] = (medH / pageHeight) * 20.0;
}

function fillItemTextFeatures(feats: Float32Array, base: number, rawText: string): void {
  const t = rawText.trim();
  if (t.length === 0) return;

  feats[base + 10] = Math.log1p(t.length) / 5.0;
  const digits = (t.match(/\d/g) || []).length;
  feats[base + 11] = digits / t.length;
  feats[base + 12] = BULLET_START.has(t[0]!) || /^\d+\.?\)?/.test(t) ? 1.0 : 0.0;
  feats[base + 13] = t.endsWith('-') ? 1.0 : 0.0;
  feats[base + 14] = SENTENCE_END.has(t[t.length - 1]!) ? 1.0 : 0.0;
  feats[base + 15] = /^[A-Z]/.test(t) ? 1.0 : 0.0;
}

export function computeLineFeatures(
  candidates: TextResult[],
  pageWidth: number,
  pageHeight: number
): Float32Array {
  const n = candidates.length;
  const feats = new Float32Array(n * 16);
  if (n === 0) return feats;

  const heights = candidates.map((c) => Math.max(1e-3, c.frame.height)).sort((a, b) => a - b);
  const medH = heights[Math.floor(heights.length / 2)] || 20;

  for (let i = 0; i < n; i++) {
    const item = candidates[i]!;
    const base = i * 16;
    fillItemGeometry(feats, base, item, pageWidth, pageHeight, medH);
    fillItemTextFeatures(feats, base, item.text || '');
  }
  return feats;
}

function computeLogPairwiseProbs(a: Float32Array | number[], n: number): Float32Array {
  const lp = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = a[i * n + j] ?? 0;
      lp[i * n + j] = -Math.log(1 + Math.exp(-v));
    }
  }
  return lp;
}

function initialSortOrder(lp: Float32Array, n: number): number[] {
  const scores: { idx: number; score: number }[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      s += (lp[i * n + j] ?? 0) - (lp[j * n + i] ?? 0);
    }
    scores.push({ idx: i, score: s });
  }
  scores.sort((x, y) => y.score - x.score);
  return scores.map((s) => s.idx);
}

function performInsertionSweep(order: number[], lp: Float32Array, n: number): boolean {
  let improved = false;
  for (const x of [...order]) {
    const rest = order.filter((idx) => idx !== x);
    let bestK = 0;
    let maxGain = -Infinity;
    for (let k = 0; k <= rest.length; k++) {
      let gain = 0;
      for (let b = 0; b < k; b++) gain += lp[rest[b]! * n + x] ?? 0;
      for (let aIdx = k; aIdx < rest.length; aIdx++) gain += lp[x * n + rest[aIdx]!] ?? 0;
      if (gain > maxGain) {
        maxGain = gain;
        bestK = k;
      }
    }
    const cur = order.indexOf(x);
    let curGain = 0;
    for (let b = 0; b < cur; b++) curGain += lp[order[b]! * n + x] ?? 0;
    for (let aIdx = cur + 1; aIdx < order.length; aIdx++) curGain += lp[x * n + order[aIdx]!] ?? 0;
    if (maxGain > curGain + 1e-5) {
      rest.splice(bestK, 0, x);
      order.splice(0, order.length, ...rest);
      improved = true;
    }
  }
  return improved;
}

export function decodeOrderFromLogits(
  a: Float32Array | number[],
  n: number,
  sweeps: number = 4
): number[] {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const lp = computeLogPairwiseProbs(a, n);
  const order = initialSortOrder(lp, n);

  for (let sw = 0; sw < sweeps; sw++) {
    const improved = performInsertionSweep(order, lp, n);
    if (!improved) break;
  }
  return order;
}

export function orderWithTransformer(
  candidates: TextResult[],
  model: RtenModel,
  pageWidth: number,
  pageHeight: number
): TextResult[] {
  if (candidates.length <= 1) return [...candidates];
  const feats = computeLineFeatures(candidates, pageWidth, pageHeight);
  const out = model.runOrder(candidates.length, feats);
  const order = decodeOrderFromLogits(out.data, candidates.length);
  return order.map((idx) => candidates[idx]!);
}

