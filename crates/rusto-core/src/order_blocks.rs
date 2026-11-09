//! Block-aware reading order decoding for order models with layout heads
//! (`same` block logits and `cls` block-class logits).

use crate::order::decode_order_logits;

const GRID_CLASSES: [usize; 3] = [5, 8, 9];
const ORDER_SWEEPS: usize = 4;

fn median_height(boxes: &[[f32; 4]]) -> f32 {
    let mut heights: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let n = heights.len();
    if n % 2 == 1 {
        heights[n / 2]
    } else {
        0.5 * (heights[n / 2 - 1] + heights[n / 2])
    }
}

/// Rows top to bottom (centres within half a line height share a row), left to right.
pub fn order_top_down(boxes: &[[f32; 4]]) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let unit = (median_height(boxes) * 0.5).max(1e-3);
    let mut keyed: Vec<(i64, f32, usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let cy = (b[1] + b[3]) / 2.0;
            ((cy / unit).round_ties_even() as i64, b[0], i)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    keyed.into_iter().map(|k| k.2).collect()
}

fn best_cut(boxes: &[[f32; 4]], vertical_axis: bool) -> (f32, Option<usize>, Vec<usize>) {
    let (lo_i, hi_i) = if vertical_axis { (1, 3) } else { (0, 2) };
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[a][lo_i].total_cmp(&boxes[b][lo_i]));
    let mut end = boxes[order[0]][hi_i];
    let mut best_gap = 0.0f32;
    let mut best_k = None;
    for k in 1..order.len() {
        let gap = boxes[order[k]][lo_i] - end;
        if gap > best_gap {
            best_gap = gap;
            best_k = Some(k);
        }
        end = end.max(boxes[order[k]][hi_i]);
    }
    (best_gap, best_k, order)
}

/// Recursive XY-cut: splits at the widest whitespace gap on either axis, the upper
/// or left part first; without a gap it falls back to top-down order.
pub fn order_xycut(boxes: &[[f32; 4]]) -> Vec<usize> {
    fn recurse(boxes: &[[f32; 4]], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let sub: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
        let (gy, ky, oy) = best_cut(&sub, true);
        let (gx, kx, ox) = best_cut(&sub, false);
        if ky.is_none() && kx.is_none() {
            return order_top_down(&sub).into_iter().map(|i| ids[i]).collect();
        }
        let (k, order) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let k = k.unwrap_or(0);
        let head: Vec<usize> = order[..k].iter().map(|&i| ids[i]).collect();
        let tail: Vec<usize> = order[k..].iter().map(|&i| ids[i]).collect();
        let mut out = recurse(boxes, head);
        out.extend(recurse(boxes, tail));
        out
    }
    recurse(boxes, (0..boxes.len()).collect())
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Average-linkage clustering of pairwise "same group" logits: merges the two groups
/// with the highest mean logit between them while that mean is positive.
pub fn cluster_same_blocks(logits: &[f32], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let neg_inf = f64::NEG_INFINITY;
    let mut s: Vec<f64> = logits[..n * n].iter().map(|&v| v as f64).collect();
    for i in 0..n {
        s[i * n + i] = 0.0;
    }
    let mut size = vec![1.0f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m = s.clone();
    for i in 0..n {
        m[i * n + i] = neg_inf;
    }
    let mut best: Vec<usize> = (0..n).map(|i| argmax(&m[i * n..(i + 1) * n])).collect();
    let mut best_v: Vec<f64> = (0..n).map(|i| m[i * n + best[i]]).collect();

    for _ in 0..n - 1 {
        let a0 = argmax(&best_v);
        let b0 = best[a0];
        if best_v[a0] <= 0.0 {
            break;
        }
        let (a, b) = (a0.min(b0), a0.max(b0));
        for k in 0..n {
            s[a * n + k] += s[b * n + k];
        }
        for k in 0..n {
            s[k * n + a] += s[k * n + b];
        }
        s[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let moved = std::mem::take(&mut members[b]);
        members[a].extend(moved);

        let row: Vec<f64> = (0..n)
            .map(|k| {
                if !alive[k] || k == a {
                    neg_inf
                } else {
                    s[a * n + k] / (size[a] * size[k])
                }
            })
            .collect();
        for k in 0..n {
            m[a * n + k] = row[k];
            m[k * n + a] = row[k];
            m[b * n + k] = neg_inf;
            m[k * n + b] = neg_inf;
        }
        best_v[b] = neg_inf;
        best[a] = argmax(&row);
        best_v[a] = row[best[a]];

        for j in 0..n {
            if j != a && alive[j] && (best[j] == a || best[j] == b) {
                best[j] = argmax(&m[j * n..(j + 1) * n]);
                best_v[j] = m[j * n + best[j]];
            }
        }
        for j in 0..n {
            if j != a && alive[j] && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }

    (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            let mut group = std::mem::take(&mut members[i]);
            group.sort_unstable();
            group
        })
        .collect()
}

fn is_grid_block(cls: &[f32], n: usize, block: &[usize]) -> bool {
    let classes = cls.len() / n.max(1);
    if classes == 0 {
        return false;
    }
    let mut sums = vec![0.0f64; classes];
    for &line in block {
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum += cls[line * classes + c] as f64;
        }
    }
    GRID_CLASSES.contains(&argmax(&sums))
}

/// Reading order with the layout heads: blocks first, then lines inside blocks.
///
/// Blocks come from clustering the same-block logits. Inside a block lines go by
/// XY-cut, or row by row in a table / form / table-of-contents block. Blocks go by
/// the mean pairwise order logit between them.
pub fn decode_blocks_order(
    a: &[f32],
    same: &[f32],
    boxes: &[[f32; 4]],
    cls: Option<&[f32]>,
) -> Vec<usize> {
    let n = boxes.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster_same_blocks(same, n);
    let inner: Vec<Vec<usize>> = blocks
        .iter()
        .map(|block| {
            let sub: Vec<[f32; 4]> = block.iter().map(|&i| boxes[i]).collect();
            let local = if cls.is_some_and(|c| is_grid_block(c, n, block)) {
                order_top_down(&sub)
            } else {
                order_xycut(&sub)
            };
            local.into_iter().map(|k| block[k]).collect()
        })
        .collect();

    let m = blocks.len();
    let mut block_logits = vec![0.0f32; m * m];
    for p in 0..m {
        for q in p + 1..m {
            let mut total = 0.0f64;
            for &i in &blocks[p] {
                for &j in &blocks[q] {
                    total += a[i * n + j] as f64;
                }
            }
            let mean = (total / (blocks[p].len() * blocks[q].len()) as f64) as f32;
            block_logits[p * m + q] = mean;
            block_logits[q * m + p] = -mean;
        }
    }
    decode_order_logits(&block_logits, m, ORDER_SWEEPS)
        .into_iter()
        .flat_map(|k| inner[k].iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_down_groups_rows() {
        let boxes = [[200.0, 0.0, 300.0, 10.0], [0.0, 1.0, 100.0, 11.0], [0.0, 30.0, 100.0, 40.0]];
        assert_eq!(order_top_down(&boxes), vec![1, 0, 2]);
    }

    #[test]
    fn test_xycut_reads_columns() {
        let boxes = [
            [0.0, 0.0, 100.0, 10.0],
            [200.0, 0.0, 300.0, 10.0],
            [0.0, 20.0, 100.0, 30.0],
            [200.0, 20.0, 300.0, 30.0],
        ];
        assert_eq!(order_xycut(&boxes), vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_cluster_merges_positive_pairs_only() {
        let n = 4;
        let mut same = vec![-5.0f32; n * n];
        for &(i, j) in &[(0, 1), (1, 0), (2, 3), (3, 2)] {
            same[i * n + j] = 4.0;
        }
        let clusters = cluster_same_blocks(&same, n);
        assert_eq!(clusters, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn test_decode_blocks_orders_blocks_by_logits() {
        let boxes = [
            [0.0, 0.0, 100.0, 10.0],
            [0.0, 20.0, 100.0, 30.0],
            [200.0, 0.0, 300.0, 10.0],
            [200.0, 20.0, 300.0, 30.0],
        ];
        let n = 4;
        let mut same = vec![-5.0f32; n * n];
        for &(i, j) in &[(0, 1), (1, 0), (2, 3), (3, 2)] {
            same[i * n + j] = 4.0;
        }
        let mut a = vec![0.0f32; n * n];
        for i in [2usize, 3] {
            for j in [0usize, 1] {
                a[i * n + j] = 6.0;
                a[j * n + i] = -6.0;
            }
        }
        let order = decode_blocks_order(&a, &same, &boxes, None);
        assert_eq!(order, vec![2, 3, 0, 1]);
    }
}

