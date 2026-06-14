/*
 * Single-hand gesture detector — a port of Python `gesture_tracker/ml_detector.py`
 * (MLHandDetector). Holds a rolling WINDOW_SIZE window of frame records, classifies
 * each full window, and fires once when the model reports beckon/shoo confidently
 * for `consecutive` windows. A fire starts a cooldown and requires a return to an
 * inert/"nothing" window before the next fire, so one motion fires once and a held
 * pose never re-fires.
 */
import { frameRecord, windowFeatures, WINDOW_SIZE } from './features';
import type { Point3 } from './features';
import type { GestureModel, GestureType } from './model';

export interface GestureFire {
  readonly gesture: GestureType;
  readonly confidence: number;
}

export class MLHandDetector {
  private readonly model: GestureModel;
  private poses: number[][] = [];
  private wrists: Array<[number, number, number]> = [];
  private streakGesture: GestureType | null = null;
  private streak = 0;
  private cooldownUntil = Number.NEGATIVE_INFINITY;
  private needIdle = false;

  /** Last classified label / probability (for the debug overlay). */
  lastLabel = 'nothing';
  lastConfidence = 0;

  constructor(model: GestureModel) {
    this.model = model;
  }

  reset(): void {
    this.poses = [];
    this.wrists = [];
    this.streakGesture = null;
    this.streak = 0;
    this.cooldownUntil = Number.NEGATIVE_INFINITY;
    this.needIdle = false;
  }

  /**
   * Feed one frame.
   * @param t timestamp in SECONDS (cooldown is configured in seconds).
   * @param landmarks 21 training-space landmarks (see orientation.ts).
   * @param isLeft whether the hand is "Left" in the training convention.
   */
  update(t: number, landmarks: ReadonlyArray<Point3>, isLeft: boolean): GestureFire | null {
    const { pose, wrist } = frameRecord(landmarks, isLeft);
    this.poses.push(pose);
    this.wrists.push(wrist);
    if (this.poses.length > WINDOW_SIZE) {
      this.poses.shift();
      this.wrists.shift();
    }
    if (this.poses.length < WINDOW_SIZE) return null;

    const feats = windowFeatures(this.poses, this.wrists);
    const { label, gesture, probability } = this.model.predict(feats);
    this.lastLabel = label;
    this.lastConfidence = probability;

    const { confidenceThreshold, consecutive, cooldownSeconds } = this.model.detector;
    const firing = gesture !== null && probability >= confidenceThreshold;

    if (!firing) {
      // An idle / low-confidence window: the hand has "returned to neutral".
      this.needIdle = false;
      this.streak = 0;
      this.streakGesture = null;
      return null;
    }

    if (gesture === this.streakGesture) {
      this.streak += 1;
    } else {
      this.streakGesture = gesture;
      this.streak = 1;
    }

    const blocked = t < this.cooldownUntil || this.needIdle;
    if (blocked || this.streak < consecutive) return null;

    // Fire.
    this.cooldownUntil = t + cooldownSeconds;
    this.needIdle = true;
    this.streak = 0;
    return { gesture: gesture as GestureType, confidence: probability };
  }
}
