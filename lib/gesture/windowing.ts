/*
 * Turn a recorded clip (sequence of per-frame records) into labeled window
 * feature vectors — the browser-side mirror of Python `training.build_windows`,
 * including the idle-relabel: a low-activity window taken from a gesture clip is
 * relabeled "nothing" so the model is not taught that a still hand is a gesture.
 */
import { POSE_DIM, WINDOW_SIZE, windowFeatures, type FrameRecord } from "./features";

export const IDLE_ACTIVITY_FLOOR = 0.06;
export const DEFAULT_STRIDE = 4;

export interface LabeledWindows {
  features: number[][];
  labels: string[];
}

function activity(
  poses: number[][],
  wrists: Array<readonly [number, number, number]>,
): number {
  // pose-shape variability
  let poseVar = 0;
  for (let j = 0; j < POSE_DIM; j++) {
    let m = 0;
    for (const p of poses) m += p[j];
    m /= poses.length;
    let v = 0;
    for (const p of poses) {
      const d = p[j] - m;
      v += d * d;
    }
    poseVar += Math.sqrt(v / poses.length);
  }
  // wrist path length (captures the toward/away push for shoo)
  let path = 0;
  for (let i = 1; i < wrists.length; i++) {
    const dx = wrists[i][0] - wrists[i - 1][0];
    const dy = wrists[i][1] - wrists[i - 1][1];
    const dz = wrists[i][2] - wrists[i - 1][2];
    path += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return poseVar + path;
}

/** Slide windows over one clip's frame records and label them. */
export function clipToWindows(
  frames: FrameRecord[],
  label: string,
  stride: number = DEFAULT_STRIDE,
): LabeledWindows {
  const features: number[][] = [];
  const labels: string[] = [];
  if (frames.length < WINDOW_SIZE) return { features, labels };

  for (let start = 0; start + WINDOW_SIZE <= frames.length; start += stride) {
    const win = frames.slice(start, start + WINDOW_SIZE);
    const poses = win.map((f) => f.pose);
    const wrists = win.map((f) => f.wrist);
    let rowLabel = label;
    if (label !== "nothing" && activity(poses, wrists) < IDLE_ACTIVITY_FLOOR) {
      rowLabel = "nothing";
    }
    features.push(windowFeatures(poses, wrists));
    labels.push(rowLabel);
  }
  return { features, labels };
}
