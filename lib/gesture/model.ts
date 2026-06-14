/*
 * Model bundle types + in-browser evaluator. Same JSON schema the Python exporter
 * and the portfolio use, so a model trained here is byte-compatible with the
 * portfolio's loader. The bundle wraps a random forest (forest.ts evaluates it).
 */
import { FEATURE_DIM, POSE_DIM, WINDOW_SIZE } from "./features";
import { forestProba, type SerializedForest } from "./forest";

export const SCHEMA_VERSION = 1;

/** Gesture a class maps to; classes absent from the map are inert ("nothing"). */
export type GestureType = "TOWARD_ME" | "TOWARD_CAMERA";

export interface DetectorParams {
  readonly confidenceThreshold: number;
  readonly consecutive: number;
  readonly cooldownSeconds: number;
  readonly resetGapSeconds: number;
  readonly minTrackingConfidence: number;
}

export const DEFAULT_DETECTOR: DetectorParams = {
  confidenceThreshold: 0.6,
  consecutive: 2,
  cooldownSeconds: 0.6,
  resetGapSeconds: 0.4,
  minTrackingConfidence: 0.5,
};

export interface BundleConfig {
  readonly classes: string[];
  readonly gestureForLabel: Record<string, GestureType>;
  readonly windowSize: number;
  readonly poseDim: number;
  readonly featureDim: number;
  readonly detector: DetectorParams;
}

export interface ModelBundle {
  readonly schema: number;
  readonly config: BundleConfig;
  readonly forest: SerializedForest;
  /** Optional metadata (held-out accuracy, counts, trained-at). */
  readonly meta?: Record<string, unknown>;
}

export interface Prediction {
  readonly label: string;
  readonly gesture: GestureType | null;
  readonly probability: number;
}

export interface GestureModel {
  readonly detector: DetectorParams;
  predict(features: number[]): Prediction;
}

/** Build a runnable model from an in-memory bundle (validates feature layout). */
export function modelFromBundle(bundle: ModelBundle): GestureModel {
  if (bundle.schema !== SCHEMA_VERSION) {
    throw new Error(`gesture model schema ${bundle.schema} != ${SCHEMA_VERSION}`);
  }
  const { config, forest } = bundle;
  if (
    config.windowSize !== WINDOW_SIZE ||
    config.poseDim !== POSE_DIM ||
    config.featureDim !== FEATURE_DIM
  ) {
    throw new Error("gesture feature layout mismatch between bundle and port");
  }
  const classes = forest.classes;
  return {
    detector: config.detector,
    predict(features: number[]): Prediction {
      const probs = forestProba(forest, features);
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      const label = classes[best];
      return {
        label,
        gesture: config.gestureForLabel[label] ?? null,
        probability: probs[best],
      };
    },
  };
}

/** Fetch + build the model; returns null on any failure (caller disables feature). */
export async function loadGestureModel(url: string): Promise<GestureModel | null> {
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) return null;
    const bundle = (await res.json()) as ModelBundle;
    return modelFromBundle(bundle);
  } catch {
    return null;
  }
}
