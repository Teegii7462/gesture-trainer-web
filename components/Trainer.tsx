"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { frameRecord, type FrameRecord, type Point3 } from "@/lib/gesture/features";
import { MLHandDetector } from "@/lib/gesture/mlDetector";
import { modelFromBundle, type GestureModel, type ModelBundle } from "@/lib/gesture/model";
import { toTrainingHand } from "@/lib/gesture/orientation";
import { clipToWindows } from "@/lib/gesture/windowing";
import { trainBundle } from "@/lib/gesture/train";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm";
const MODEL_ASSET =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const LABEL_TEXT: Record<string, string> = {
  TOWARD_ME: "TOWARD ME (beckon)",
  TOWARD_CAMERA: "TOWARD CAMERA (shoo)",
};
const CLIP_SECONDS = 2;

interface Stats {
  contributions: number;
  windows: number;
  counts: Record<string, number>;
  hasModel: boolean;
}

type RecState = { label: string; frames: FrameRecord[]; until: number } | null;

export default function Trainer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const detectorRef = useRef<MLHandDetector | null>(null);
  const recRef = useRef<RecState>(null);
  const pendingRef = useRef<{ features: number[][]; labels: string[] }>({ features: [], labels: [] });
  const flashRef = useRef<{ text: string; at: number } | null>(null);
  const rafRef = useRef<number>(0);
  const handPresentRef = useRef(false);

  const [status, setStatus] = useState("Loading camera + model…");
  const [ready, setReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [handPresent, setHandPresent] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [pending, setPending] = useState<Record<string, number>>({ beckon: 0, shoo: 0, nothing: 0 });
  const [recLabel, setRecLabel] = useState<string | null>(null);
  const [recRemaining, setRecRemaining] = useState(0);
  const [busy, setBusy] = useState(false);
  const [accuracy, setAccuracy] = useState<number | null>(null);

  const refreshStats = useCallback(async () => {
    try {
      const s = (await fetch("/api/stats", { cache: "no-store" }).then((r) => r.json())) as Stats;
      setStats(s);
    } catch {
      /* ignore */
    }
  }, []);

  const loadModel = useCallback(async () => {
    try {
      const res = await fetch("/api/model", { cache: "no-store" });
      if (!res.ok) {
        setModelReady(false);
        return;
      }
      const bundle = (await res.json()) as ModelBundle;
      const model: GestureModel = modelFromBundle(bundle);
      detectorRef.current = new MLHandDetector(model);
      const acc = (bundle.meta?.heldoutAccuracy as number | null) ?? null;
      setAccuracy(typeof acc === "number" ? acc : null);
      setModelReady(true);
    } catch {
      setModelReady(false);
    }
  }, []);

  // ---- main loop ---------------------------------------------------------
  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);

    let result: HandLandmarkerResult | null = null;
    try {
      result = landmarker.detectForVideo(video, performance.now());
    } catch {
      result = null;
    }

    const hasHand = !!result && result.landmarks.length > 0;
    const nowSec = performance.now() / 1000;

    if (hasHand && result) {
      const lm = result.landmarks[0];
      const side = result.handedness[0]?.[0]?.categoryName ?? "Right";
      // draw skeleton (canvas is CSS-mirrored to match the selfie video)
      drawHand(ctx, lm, w, h);

      const live = toTrainingHand(lm as unknown as { x: number; y: number; z: number }[], side);
      const points = live.points as Point3[];

      // recording?
      const rec = recRef.current;
      if (rec) {
        rec.frames.push(frameRecord(points, live.isLeft));
      } else if (detectorRef.current) {
        const fire = detectorRef.current.update(nowSec, points, live.isLeft);
        if (fire) flashRef.current = { text: LABEL_TEXT[fire.gesture] ?? fire.gesture, at: nowSec };
      }
    } else if (detectorRef.current) {
      detectorRef.current.reset();
    }

    if (handPresentRef.current !== hasHand) {
      handPresentRef.current = hasHand;
      setHandPresent(hasHand);
    }

    // recording countdown / finalize
    const rec = recRef.current;
    if (rec) {
      const remaining = Math.max(0, rec.until - nowSec);
      setRecRemaining(Math.round(remaining * 10) / 10);
      if (nowSec >= rec.until) finalizeClip(rec);
    }

    // flash banner
    const flash = flashRef.current;
    if (flash && nowSec - flash.at < 1.0) {
      ctx.save();
      ctx.scale(-1, 1); // text un-mirror (canvas drawn in mirrored space)
      ctx.font = "bold 34px -apple-system, system-ui, sans-serif";
      ctx.fillStyle = flash.text.includes("TOWARD ME") ? "#5fe07a" : "#5fadff";
      ctx.textAlign = "center";
      ctx.fillText(flash.text, -w / 2, 52);
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(loop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const finalizeClip = useCallback(
    (rec: NonNullable<RecState>) => {
      recRef.current = null;
      setRecLabel(null);
      const { features, labels } = clipToWindows(rec.frames, rec.label);
      if (features.length) {
        pendingRef.current.features.push(...features);
        pendingRef.current.labels.push(...labels);
        const counts: Record<string, number> = { beckon: 0, shoo: 0, nothing: 0 };
        for (const l of pendingRef.current.labels) counts[l] = (counts[l] ?? 0) + 1;
        setPending(counts);
      }
    },
    [],
  );

  // ---- setup -------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
        const landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_ASSET, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 1,
        });
        if (cancelled) return;
        landmarkerRef.current = landmarker;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 960, height: 720, facingMode: "user" },
        });
        if (cancelled) return;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        await loadModel();
        await refreshStats();
        if (cancelled) return;
        setReady(true);
        setStatus("");
        rafRef.current = requestAnimationFrame(loop);
      } catch (err) {
        setStatus(
          "Could not start camera/model: " +
            (err instanceof Error ? err.message : String(err)) +
            ". Allow camera access and reload.",
        );
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      const v = videoRef.current;
      const s = v?.srcObject as MediaStream | null;
      s?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- actions -----------------------------------------------------------
  const record = (label: string) => {
    if (recRef.current || busy) return;
    recRef.current = { label, frames: [], until: performance.now() / 1000 + CLIP_SECONDS };
    setRecLabel(label);
  };

  const contribute = async () => {
    const payload = pendingRef.current;
    if (!payload.features.length || busy) return;
    setBusy(true);
    setStatus("Uploading your samples…");
    try {
      const res = await fetch("/api/contribute", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.ok) {
        pendingRef.current = { features: [], labels: [] };
        setPending({ beckon: 0, shoo: 0, nothing: 0 });
        setStats(data.stats as Stats);
        setStatus("Thanks — samples added to the shared dataset.");
      } else {
        setStatus("Upload failed: " + data.error);
      }
    } catch (e) {
      setStatus("Upload failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setBusy(false);
    }
  };

  const train = async () => {
    if (busy) return;
    setBusy(true);
    setStatus("Fetching shared dataset…");
    try {
      const { X, y } = (await fetch("/api/dataset", { cache: "no-store" }).then((r) => r.json())) as {
        X: number[][];
        y: string[];
      };
      if (!X?.length) {
        setStatus("No samples yet — record and contribute first.");
        setBusy(false);
        return;
      }
      setStatus(`Training on ${X.length} windows…`);
      await new Promise((r) => setTimeout(r, 30)); // let the UI paint
      const { bundle, report } = trainBundle(X, y);
      setStatus("Saving global model…");
      const res = await fetch("/api/model", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(bundle),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "save failed");
      detectorRef.current = new MLHandDetector(modelFromBundle(bundle));
      setModelReady(true);
      setAccuracy(report.heldoutAccuracy);
      await refreshStats();
      setStatus(
        report.heldoutAccuracy != null
          ? `Trained! Held-out accuracy ${(report.heldoutAccuracy * 100).toFixed(0)}% on ${report.nWindows} windows. Try gesturing.`
          : `Trained on ${report.nWindows} windows. Try gesturing.`,
      );
    } catch (e) {
      setStatus("Training failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setBusy(false);
    }
  };

  const pendingTotal = pending.beckon + pending.shoo + pending.nothing;

  return (
    <main className="wrap">
      <h1>Gesture Trainer</h1>

      <div className="stage">
        <video ref={videoRef} className="mirror" playsInline muted />
        <canvas ref={canvasRef} className="mirror overlay" />
        {!ready && <div className="loading">{status || "Starting…"}</div>}
        <div className="badge">
          {handPresent ? "✋ hand detected" : "no hand in frame"}
          {recLabel && <span className="rec"> ● REC {recLabel.toUpperCase()} {recRemaining}s</span>}
        </div>
      </div>

      <section className="panel">
        <div className="row">
          <button className="b beckon" disabled={!ready || busy || !!recLabel} onClick={() => record("beckon")}>● Record BECKON</button>
          <button className="b shoo" disabled={!ready || busy || !!recLabel} onClick={() => record("shoo")}>● Record SHOO</button>
          <button className="b nothing" disabled={!ready || busy || !!recLabel} onClick={() => record("nothing")}>● Record NOTHING</button>
        </div>
        <div className="row small">
          <span className="pill">pending: beckon {pending.beckon} · shoo {pending.shoo} · nothing {pending.nothing}</span>
          <button className="b ghost" disabled={!pendingTotal || busy} onClick={contribute}>⬆ Contribute {pendingTotal} windows</button>
        </div>
        <div className="row">
          <button className="b train" disabled={busy} onClick={train}>⟳ Train global model</button>
          <span className="pill">
            global: {stats ? `${stats.windows} windows · ${stats.contributions} contributions` : "…"}
            {modelReady && accuracy != null && ` · model ${(accuracy * 100).toFixed(0)}%`}
            {!modelReady && " · no model yet"}
          </span>
        </div>
        <p className="status">{status || (modelReady ? "Model loaded — gesture at the camera." : "Record samples, contribute, then train.")}</p>
        <p className="hint">
          <b>Beckon</b> = curl fingers in (&ldquo;come here&rdquo;). <b>Shoo</b> = push open palm out (&ldquo;go away&rdquo;).
          <b> Nothing</b> = hold still, rest, type, wave hello. Record a few of each, contribute, and train.
        </p>
      </section>
    </main>
  );
}

function drawHand(
  ctx: CanvasRenderingContext2D,
  lm: ReadonlyArray<{ x: number; y: number }>,
  w: number,
  h: number,
) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(120,120,130,0.9)";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lm[a].x * w, lm[a].y * h);
    ctx.lineTo(lm[b].x * w, lm[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#8ed6ff";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}
