/*
 * Feature extraction for the trained beckon/shoo classifier — a faithful port of
 * the Python `gesture_tracker/features.py`. The website runs the SAME pipeline in
 * the browser so a model trained on the user's hand can be evaluated client-side
 * (no server). The two implementations must agree to ~1e-6: `gesture.test.ts`
 * checks this port against golden fixtures emitted by the Python exporter, so any
 * drift fails CI.
 *
 * Two stages, exactly as in Python:
 *  1. frameRecord — one hand's 21 landmarks -> a translation/scale/handedness-
 *     invariant pose vector (POSE_DIM=20), plus the raw wrist used for window
 *     motion.
 *  2. windowFeatures — aggregate a rolling window of frame records (mean, std,
 *     net change, peak velocity per pose dim, plus wrist displacement/path) into
 *     the model's input vector (FEATURE_DIM=85).
 */

/** A single landmark: MediaPipe normalized image coords + relative depth. */
export type Point3 = readonly [number, number, number];

export const WINDOW_SIZE = 16; // frames per classification window (~0.5s @ 30fps)

const WRIST = 0;
// (mcp, pip, dip, tip) per non-thumb finger.
const FINGERS: ReadonlyArray<readonly [number, number, number, number]> = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const FINGERTIPS = [8, 12, 16, 20] as const;
const INDEX_MCP = 5;
const PINKY_MCP = 17;
const MIDDLE_MCP = 9;
const EPS = 1e-8;

/** Length of the per-frame pose vector: curls(4) + mean(1) + normal(3) + tips(12). */
export const POSE_DIM = 4 + 1 + 3 + FINGERTIPS.length * 3; // 20
/** Length of the window feature vector: 4*POSE_DIM + disp(3) + [path_len, peak_speed]. */
export const FEATURE_DIM = 4 * POSE_DIM + 5; // 85

export interface FrameRecord {
  /** Pose vector (length POSE_DIM). */
  readonly pose: number[];
  /** Raw wrist xyz (un-centered) for window-level motion. */
  readonly wrist: [number, number, number];
}

const sub = (a: Point3, b: Point3): [number, number, number] => [
  a[0] - b[0],
  a[1] - b[1],
  a[2] - b[2],
];

const norm3 = (v: readonly number[]): number =>
  Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

const cross = (
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): [number, number, number] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/**
 * Wrist-center, scale-normalize, and mirror left hands to one chirality.
 * Mirrors Python `_canonical`. Returns the centered points and the ORIGINAL
 * (un-centered, un-mirrored) wrist used for window motion.
 */
function canonical(
  landmarks: ReadonlyArray<Point3>,
  isLeft: boolean,
): { points: [number, number, number][]; wrist: [number, number, number] } {
  const wrist0 = landmarks[WRIST];
  const wrist: [number, number, number] = [wrist0[0], wrist0[1], wrist0[2]];

  const centered: [number, number, number][] = landmarks.map((p) => [
    p[0] - wrist[0],
    p[1] - wrist[1],
    p[2] - wrist[2],
  ]);

  let scale = norm3(centered[MIDDLE_MCP]); // wrist->middle-MCP length
  if (scale < EPS) scale = 1.0;
  for (const p of centered) {
    p[0] /= scale;
    p[1] /= scale;
    p[2] /= scale;
  }
  if (isLeft) {
    for (const p of centered) p[0] = -p[0]; // one shared chirality
  }
  return { points: centered, wrist };
}

/** Normalized bend angle (0..1) between segments a->b and b->c. Mirrors `_bend`. */
function bend(a: Point3, b: Point3, c: Point3): number {
  const v1 = sub(b, a);
  const v2 = sub(c, b);
  const n1 = norm3(v1);
  const n2 = norm3(v2);
  if (n1 < EPS || n2 < EPS) return 0;
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const cos = Math.max(-1, Math.min(1, dot / (n1 * n2)));
  return Math.acos(cos) / Math.PI;
}

/** Per-frame pose vector (length POSE_DIM). Mirrors `_pose_vector`. */
function poseVector(pts: ReadonlyArray<Point3>): number[] {
  const curls = FINGERS.map(([mcp, pip, , tip]) => bend(pts[mcp], pts[pip], pts[tip]));
  const meanCurl = (curls[0] + curls[1] + curls[2] + curls[3]) / 4;

  const v1 = sub(pts[INDEX_MCP], pts[WRIST]);
  const v2 = sub(pts[PINKY_MCP], pts[WRIST]);
  const raw = cross(v1, v2);
  const nn = norm3(raw);
  const normal = nn > EPS ? [raw[0] / nn, raw[1] / nn, raw[2] / nn] : [0, 0, 0];

  const tips: number[] = [];
  for (const tip of FINGERTIPS) tips.push(pts[tip][0], pts[tip][1], pts[tip][2]);

  return [...curls, meanCurl, ...normal, ...tips];
}

/** One frame's record: pose vector + raw wrist. Mirrors `frame_record`. */
export function frameRecord(landmarks: ReadonlyArray<Point3>, isLeft: boolean): FrameRecord {
  const { points, wrist } = canonical(landmarks, isLeft);
  return { pose: poseVector(points), wrist };
}

const mean = (cols: number[][], dim: number): number[] => {
  const out = new Array<number>(dim).fill(0);
  for (const row of cols) for (let j = 0; j < dim; j++) out[j] += row[j];
  for (let j = 0; j < dim; j++) out[j] /= cols.length;
  return out;
};

// Population standard deviation (numpy default ddof=0), matching `poses.std(axis=0)`.
const std = (cols: number[][], dim: number, means: number[]): number[] => {
  const out = new Array<number>(dim).fill(0);
  for (const row of cols) {
    for (let j = 0; j < dim; j++) {
      const d = row[j] - means[j];
      out[j] += d * d;
    }
  }
  for (let j = 0; j < dim; j++) out[j] = Math.sqrt(out[j] / cols.length);
  return out;
};

/**
 * Aggregate a window of frame records into the model input vector.
 * Mirrors Python `window_features`. `poses` is (W x POSE_DIM), `wrists` is (W x 3).
 */
export function windowFeatures(
  poses: number[][],
  wrists: Array<readonly [number, number, number]>,
): number[] {
  const w = poses.length;
  const m = mean(poses, POSE_DIM);
  const s = std(poses, POSE_DIM, m);

  const net = new Array<number>(POSE_DIM);
  for (let j = 0; j < POSE_DIM; j++) net[j] = poses[w - 1][j] - poses[0][j];

  // peak per-frame |delta| of each pose dim (np.abs(np.diff(...)).max(axis=0)).
  const peakVel = new Array<number>(POSE_DIM).fill(0);
  for (let i = 1; i < w; i++) {
    for (let j = 0; j < POSE_DIM; j++) {
      const d = Math.abs(poses[i][j] - poses[i - 1][j]);
      if (d > peakVel[j]) peakVel[j] = d;
    }
  }

  // wrist motion: displacement, path length, peak step speed.
  const disp: [number, number, number] = [
    wrists[w - 1][0] - wrists[0][0],
    wrists[w - 1][1] - wrists[0][1],
    wrists[w - 1][2] - wrists[0][2],
  ];
  let pathLen = 0;
  let peakSpeed = 0;
  for (let i = 1; i < w; i++) {
    const step = norm3(sub(wrists[i], wrists[i - 1]));
    pathLen += step;
    if (step > peakSpeed) peakSpeed = step;
  }

  return [...m, ...s, ...net, ...peakVel, ...disp, pathLen, peakSpeed];
}
