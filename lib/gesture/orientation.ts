/*
 * Live-camera input adapter.
 *
 * The model is trained on a SELFIE-MIRRORED feed: the Python recorder
 * (`webcam_server.py` / `demo.py`) does `cv2.flip(frame, 1)` before MediaPipe, so
 * the recorded landmarks have x mirrored and the user's right hand is reported as
 * "Left". The website feeds MediaPipe the RAW camera, so to match training we
 * mirror x (x -> 1 - x) and swap the handedness label here, at the boundary.
 *
 * This is the one part of the pipeline that can only be fully confirmed with a
 * live camera. If gestures ever read inverted or unreliable, flip MIRROR_INPUT —
 * it is intentionally a single switch. Everything downstream (features, forest,
 * detector) is verified against Python by the golden-fixture test.
 */
import type { Point3 } from './features';

/** Mirror live input into the model's training convention. See file header. */
export const MIRROR_INPUT = true;

export interface LiveHand {
  /** 21 landmarks, training-space, ready for frameRecord. */
  readonly points: Point3[];
  /** Whether this hand is "Left" in the model's (training) convention. */
  readonly isLeft: boolean;
}

/** A MediaPipe normalized landmark. */
export interface MpLandmark {
  x: number;
  y: number;
  z: number;
}

/**
 * Convert one MediaPipe hand (raw camera) into the training convention.
 * @param landmarks MediaPipe normalized landmarks for one hand (length 21).
 * @param mpHandedness MediaPipe's reported category name: 'Left' | 'Right'.
 */
export function toTrainingHand(
  landmarks: ReadonlyArray<MpLandmark>,
  mpHandedness: string,
): LiveHand {
  if (MIRROR_INPUT) {
    const points: Point3[] = landmarks.map((p) => [1 - p.x, p.y, p.z]);
    // Selfie flip inverts the reported side: raw 'Right' == training 'Left'.
    return { points, isLeft: mpHandedness === 'Right' };
  }
  const points: Point3[] = landmarks.map((p) => [p.x, p.y, p.z]);
  return { points, isLeft: mpHandedness === 'Left' };
}
