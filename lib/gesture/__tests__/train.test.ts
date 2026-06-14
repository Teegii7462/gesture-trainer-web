import { describe, expect, it } from "vitest";
import { FEATURE_DIM, POSE_DIM, WINDOW_SIZE, type FrameRecord } from "../features";
import { trainForest } from "../forestTrainer";
import { trainBundle } from "../train";
import { modelFromBundle } from "../model";
import { clipToWindows } from "../windowing";

// Deterministic RNG for synthetic data.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Three separable Gaussian clusters in FEATURE_DIM space, one per class. */
function makeDataset(n: number, seed = 1) {
  const r = rng(seed);
  const gauss = () => {
    // Box-Muller
    const u = Math.max(1e-9, r());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
  const labels = ["nothing", "beckon", "shoo"];
  const centers: Record<string, number[]> = {
    nothing: Array.from({ length: FEATURE_DIM }, (_, i) => (i < 5 ? 0 : 0)),
    beckon: Array.from({ length: FEATURE_DIM }, (_, i) => (i < 5 ? 3 : 0)),
    shoo: Array.from({ length: FEATURE_DIM }, (_, i) => (i >= 80 ? 3 : 0)),
  };
  const X: number[][] = [];
  const y: string[] = [];
  for (let k = 0; k < n; k++) {
    for (const lbl of labels) {
      X.push(centers[lbl].map((c) => c + gauss() * 0.5));
      y.push(lbl);
    }
  }
  return { X, y };
}

describe("forest trainer", () => {
  it("learns separable classes and predicts held-out points", () => {
    const { X, y } = makeDataset(60, 1);
    const { X: Xte, y: yte } = makeDataset(20, 999);
    const forest = trainForest(X, y, { nEstimators: 40, seed: 3 });
    expect(forest.classes).toEqual(["beckon", "nothing", "shoo"]);

    // evaluate via the same evaluator the app uses
    const bundleModel = modelFromBundle({
      schema: 1,
      config: {
        classes: forest.classes,
        gestureForLabel: { beckon: "TOWARD_ME", shoo: "TOWARD_CAMERA" },
        windowSize: WINDOW_SIZE,
        poseDim: POSE_DIM,
        featureDim: FEATURE_DIM,
        detector: {
          confidenceThreshold: 0.6,
          consecutive: 2,
          cooldownSeconds: 0.6,
          resetGapSeconds: 0.4,
          minTrackingConfidence: 0.5,
        },
      },
      forest,
    });
    let correct = 0;
    for (let i = 0; i < Xte.length; i++) {
      if (bundleModel.predict(Xte[i]).label === yte[i]) correct++;
    }
    expect(correct / Xte.length).toBeGreaterThan(0.9);
  });
});

describe("trainBundle", () => {
  it("produces a valid bundle with held-out accuracy and gesture mapping", () => {
    const { X, y } = makeDataset(50, 7);
    const { bundle, report } = trainBundle(X, y, { nEstimators: 40 });
    expect(bundle.config.featureDim).toBe(FEATURE_DIM);
    expect(bundle.config.gestureForLabel.beckon).toBe("TOWARD_ME");
    expect(report.heldoutAccuracy).not.toBeNull();
    expect(report.heldoutAccuracy!).toBeGreaterThan(0.9);

    const model = modelFromBundle(bundle);
    const beckonish = Array.from({ length: FEATURE_DIM }, (_, i) => (i < 5 ? 3 : 0));
    const pred = model.predict(beckonish);
    expect(pred.label).toBe("beckon");
    expect(pred.gesture).toBe("TOWARD_ME");
  });
});

describe("clipToWindows", () => {
  const flat = (v: number) =>
    ({ pose: new Array(POSE_DIM).fill(v), wrist: [0, 0, 0] }) as FrameRecord;

  it("relabels a static gesture clip as nothing", () => {
    const frames: FrameRecord[] = Array.from({ length: 20 }, () => flat(0.3));
    const { labels } = clipToWindows(frames, "beckon");
    expect(labels.length).toBeGreaterThan(0);
    expect(labels.every((l) => l === "nothing")).toBe(true);
  });

  it("keeps the label for a moving clip and emits FEATURE_DIM vectors", () => {
    const frames: FrameRecord[] = Array.from({ length: 20 }, (_, i) => ({
      pose: new Array(POSE_DIM).fill(0).map((_, j) => (j === 0 ? i * 0.1 : 0)),
      wrist: [i * 0.05, 0, -i * 0.05],
    }));
    const { features, labels } = clipToWindows(frames, "shoo");
    expect(features.length).toBeGreaterThan(0);
    expect(features[0].length).toBe(FEATURE_DIM);
    expect(labels).toContain("shoo");
  });

  it("returns nothing for clips shorter than the window", () => {
    const { features } = clipToWindows([flat(0.3)], "beckon");
    expect(features.length).toBe(0);
  });
});
