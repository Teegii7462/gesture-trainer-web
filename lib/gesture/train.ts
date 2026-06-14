/*
 * Train + evaluate + bundle. Given the shared dataset (window feature vectors +
 * labels), trains a random forest, measures held-out accuracy, then retrains on
 * all data and wraps it in a ModelBundle ready to store and serve.
 */
import { FEATURE_DIM, POSE_DIM, WINDOW_SIZE } from "./features";
import { forestProba } from "./forest";
import { trainForest, type TrainOptions } from "./forestTrainer";
import {
  DEFAULT_DETECTOR,
  SCHEMA_VERSION,
  type GestureType,
  type ModelBundle,
} from "./model";

export const LABELS = ["nothing", "beckon", "shoo"] as const;
export const GESTURE_FOR_LABEL: Record<string, GestureType> = {
  beckon: "TOWARD_ME",
  shoo: "TOWARD_CAMERA",
};
/** Keep in-browser training snappy: subsample very large shared datasets. */
const MAX_TRAIN_ROWS = 8000;

export interface TrainReport {
  nWindows: number;
  classCounts: Record<string, number>;
  heldoutAccuracy: number | null;
  testN: number;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Stratified train/test split by label. */
function split(
  X: number[][],
  y: string[],
  testFrac: number,
  rand: () => number,
): { Xtr: number[][]; ytr: string[]; Xte: number[][]; yte: string[] } {
  const byLabel = new Map<string, number[]>();
  y.forEach((l, i) => {
    if (!byLabel.has(l)) byLabel.set(l, []);
    byLabel.get(l)!.push(i);
  });
  const test = new Set<number>();
  for (const idxs of byLabel.values()) {
    // shuffle
    for (let i = idxs.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idxs[i], idxs[j]] = [idxs[j], idxs[i]];
    }
    const nTest = Math.floor(idxs.length * testFrac);
    for (let i = 0; i < nTest; i++) test.add(idxs[i]);
  }
  const Xtr: number[][] = [];
  const ytr: string[] = [];
  const Xte: number[][] = [];
  const yte: string[] = [];
  for (let i = 0; i < X.length; i++) {
    if (test.has(i)) {
      Xte.push(X[i]);
      yte.push(y[i]);
    } else {
      Xtr.push(X[i]);
      ytr.push(y[i]);
    }
  }
  return { Xtr, ytr, Xte, yte };
}

function predictLabel(
  forest: ReturnType<typeof trainForest>,
  x: number[],
): string {
  const probs = forestProba(forest, x);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return forest.classes[best];
}

function subsample(
  X: number[][],
  y: string[],
  max: number,
  rand: () => number,
): { X: number[][]; y: string[] } {
  if (X.length <= max) return { X, y };
  const idx = Array.from({ length: X.length }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const keep = idx.slice(0, max);
  return { X: keep.map((i) => X[i]), y: keep.map((i) => y[i]) };
}

export interface TrainResult {
  bundle: ModelBundle;
  report: TrainReport;
}

/** Train a global model from the full dataset. */
export function trainBundle(
  Xall: number[][],
  yall: string[],
  options: Partial<TrainOptions> = {},
): TrainResult {
  if (Xall.length === 0) throw new Error("no training windows");
  const rand = mulberry32(7);
  const { X, y } = subsample(Xall, yall, MAX_TRAIN_ROWS, rand);

  const classCounts: Record<string, number> = {};
  for (const l of y) classCounts[l] = (classCounts[l] ?? 0) + 1;

  // held-out evaluation when feasible
  const labelsPresent = Object.keys(classCounts);
  const minClass = Math.min(...Object.values(classCounts));
  let heldoutAccuracy: number | null = null;
  let testN = 0;
  if (labelsPresent.length >= 2 && minClass >= 4 && X.length >= 16) {
    const { Xtr, ytr, Xte, yte } = split(X, y, 0.25, rand);
    if (Xte.length > 0 && new Set(ytr).size >= 2) {
      const evalForest = trainForest(Xtr, ytr, options);
      let correct = 0;
      for (let i = 0; i < Xte.length; i++) {
        if (predictLabel(evalForest, Xte[i]) === yte[i]) correct++;
      }
      heldoutAccuracy = correct / Xte.length;
      testN = Xte.length;
    }
  }

  const forest = trainForest(X, y, options);
  const bundle: ModelBundle = {
    schema: SCHEMA_VERSION,
    config: {
      classes: forest.classes,
      gestureForLabel: GESTURE_FOR_LABEL,
      windowSize: WINDOW_SIZE,
      poseDim: POSE_DIM,
      featureDim: FEATURE_DIM,
      detector: DEFAULT_DETECTOR,
    },
    forest,
    meta: {
      trainedAt: new Date().toISOString(),
      nWindows: Xall.length,
      classCounts,
      heldoutAccuracy,
    },
  };
  return {
    bundle,
    report: { nWindows: Xall.length, classCounts, heldoutAccuracy, testN },
  };
}
