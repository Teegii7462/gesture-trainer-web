/*
 * In-browser random-forest trainer. Produces a SerializedForest (the exact flat
 * format forest.ts evaluates), so a model trained here runs through the same
 * inference path as the Python-exported one. CART trees with Gini splits,
 * bootstrap bagging, and per-node feature subsampling — a compact JS mirror of
 * scikit-learn's RandomForestClassifier (predict_proba = mean leaf distribution).
 */
import type { SerializedForest, SerializedTree } from "./forest";

export interface TrainOptions {
  nEstimators: number;
  maxDepth: number;
  minSamplesLeaf: number;
  minSamplesSplit: number;
  /** Candidate features per split (default round(sqrt(nFeatures))). */
  maxFeatures?: number;
  /** Cap candidate thresholds per feature for speed (quantile-sampled). */
  maxThresholds: number;
  seed: number;
}

export const DEFAULT_TRAIN: TrainOptions = {
  nEstimators: 60,
  maxDepth: 16,
  minSamplesLeaf: 2,
  minSamplesSplit: 4,
  maxThresholds: 24,
  seed: 1,
};

/** Deterministic RNG (mulberry32) so training is reproducible. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Node {
  feature: number;
  threshold: number;
  left: number;
  right: number;
  value: number[]; // normalized class distribution
}

function classDistribution(
  rows: number[],
  y: number[],
  nClasses: number,
): number[] {
  const counts = new Array<number>(nClasses).fill(0);
  for (const i of rows) counts[y[i]]++;
  const total = rows.length || 1;
  return counts.map((c) => c / total);
}

function gini(dist: number[]): number {
  let s = 0;
  for (const p of dist) s += p * p;
  return 1 - s;
}

function candidateThresholds(
  values: number[],
  maxThresholds: number,
  rand: () => number,
): number[] {
  const sorted = Array.from(new Set(values)).sort((a, b) => a - b);
  if (sorted.length <= 1) return [];
  const mids: number[] = [];
  for (let i = 1; i < sorted.length; i++) mids.push((sorted[i - 1] + sorted[i]) / 2);
  if (mids.length <= maxThresholds) return mids;
  // quantile-sample candidate thresholds for speed
  const out: number[] = [];
  for (let k = 0; k < maxThresholds; k++) {
    out.push(mids[Math.floor((k + rand() * 0.0) / maxThresholds * mids.length)]);
  }
  return Array.from(new Set(out));
}

function buildTree(
  X: number[][],
  y: number[],
  rows: number[],
  nClasses: number,
  nFeatures: number,
  maxFeatures: number,
  opts: TrainOptions,
  rand: () => number,
): SerializedTree {
  const nodes: Node[] = [];

  function recurse(rs: number[], depth: number): number {
    const dist = classDistribution(rs, y, nClasses);
    const idx = nodes.length;
    nodes.push({ feature: -2, threshold: 0, left: -1, right: -1, value: dist });

    const impurity = gini(dist);
    const pure = impurity <= 1e-9;
    if (
      pure ||
      depth >= opts.maxDepth ||
      rs.length < opts.minSamplesSplit ||
      rs.length < 2 * opts.minSamplesLeaf
    ) {
      return idx; // leaf
    }

    // pick a random feature subset
    const feats: number[] = [];
    const pool = Array.from({ length: nFeatures }, (_, i) => i);
    for (let k = 0; k < maxFeatures && pool.length; k++) {
      const j = Math.floor(rand() * pool.length);
      feats.push(pool[j]);
      pool.splice(j, 1);
    }

    let bestGain = 0;
    let bestFeat = -1;
    let bestThr = 0;
    let bestLeft: number[] = [];
    let bestRight: number[] = [];
    const parentImp = impurity;
    const n = rs.length;

    for (const f of feats) {
      const vals = rs.map((i) => X[i][f]);
      for (const thr of candidateThresholds(vals, opts.maxThresholds, rand)) {
        const left: number[] = [];
        const right: number[] = [];
        for (const i of rs) (X[i][f] <= thr ? left : right).push(i);
        if (left.length < opts.minSamplesLeaf || right.length < opts.minSamplesLeaf) {
          continue;
        }
        const gl = gini(classDistribution(left, y, nClasses));
        const gr = gini(classDistribution(right, y, nClasses));
        const weighted = (left.length / n) * gl + (right.length / n) * gr;
        const gain = parentImp - weighted;
        if (gain > bestGain) {
          bestGain = gain;
          bestFeat = f;
          bestThr = thr;
          bestLeft = left;
          bestRight = right;
        }
      }
    }

    if (bestFeat === -1 || bestGain <= 1e-9) return idx; // no useful split -> leaf

    const leftIdx = recurse(bestLeft, depth + 1);
    const rightIdx = recurse(bestRight, depth + 1);
    nodes[idx].feature = bestFeat;
    nodes[idx].threshold = bestThr;
    nodes[idx].left = leftIdx;
    nodes[idx].right = rightIdx;
    return idx;
  }

  recurse(rows, 0);
  return {
    feature: nodes.map((n) => n.feature),
    threshold: nodes.map((n) => n.threshold),
    left: nodes.map((n) => n.left),
    right: nodes.map((n) => n.right),
    value: nodes.map((n) => n.value),
  };
}

/** Train a random forest. `y` are string labels; `classes` is sorted unique. */
export function trainForest(
  X: number[][],
  yLabels: string[],
  options: Partial<TrainOptions> = {},
): SerializedForest {
  const opts = { ...DEFAULT_TRAIN, ...options };
  if (X.length === 0) throw new Error("no training rows");
  const nFeatures = X[0].length;
  const classes = Array.from(new Set(yLabels)).sort();
  const classIndex = new Map(classes.map((c, i) => [c, i]));
  const y = yLabels.map((l) => classIndex.get(l)!);
  const nClasses = classes.length;
  const maxFeatures =
    opts.maxFeatures ?? Math.max(1, Math.round(Math.sqrt(nFeatures)));
  const rand = rng(opts.seed);

  const trees: SerializedTree[] = [];
  const n = X.length;
  for (let t = 0; t < opts.nEstimators; t++) {
    const rows = new Array<number>(n);
    for (let i = 0; i < n; i++) rows[i] = Math.floor(rand() * n); // bootstrap
    trees.push(buildTree(X, y, rows, nClasses, nFeatures, maxFeatures, opts, rand));
  }

  return { classes, n_estimators: trees.length, trees };
}
