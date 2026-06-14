# Gesture Trainer (web) — shared beckon / shoo model

A browser app where anyone can **record hand gestures, contribute them, and train a
shared model** that everyone then uses live. No Python, no server-side ML — the
whole pipeline runs in the browser:

webcam → MediaPipe Hand Landmarker → feature windows → in-browser random-forest
training → live inference.

Companion to the Python project
[`hand-gesture-tracker`](https://github.com/Teegii7462/hand-gesture-tracker); the
feature pipeline (`lib/gesture/features.ts`) is a faithful port of that repo's
`features.py`, so models are compatible.

## How it works

- **Record** clips of `beckon` (curl in), `shoo` (palm out), and `nothing` (rest,
  fidget, wave hello). Each clip is windowed into 85-dim feature vectors in the
  browser; a low-motion window from a gesture clip is auto-relabeled `nothing`.
- **Contribute** uploads only the numeric feature windows (never video/images) to
  shared storage.
- **Train global model** pulls the shared dataset, trains a random forest **in your
  browser**, reports held-out accuracy, and saves it as the new global model that
  every visitor loads.
- **Live**: the loaded model classifies a rolling window and fires `TOWARD_ME` /
  `TOWARD_CAMERA` with a debounce (one fire per gesture; a still hand stays quiet).

Privacy: the camera stream stays on-device. Only derived feature vectors leave the
browser.

## Architecture

```
app/
  page.tsx                  -> renders <Trainer/>
  api/contribute/route.ts   POST feature windows -> shared store
  api/dataset/route.ts      GET  all windows (for in-browser training)
  api/model/route.ts        GET/POST the global model bundle
  api/stats/route.ts        GET  dataset counts (cheap, from blob names)
components/Trainer.tsx       webcam + MediaPipe + record/contribute/train/live
lib/gesture/                 features, forest (eval), forestTrainer (train),
                             windowing, train (bundle+holdout), model, mlDetector
lib/storage.ts              Vercel Blob (prod) / .data files (local dev)
```

## Local dev

```bash
npm install
npm run dev          # http://localhost:3000  (uses .data/ files for storage)
npm test             # vitest: trainer + windowing
npm run build
```

Without `BLOB_READ_WRITE_TOKEN`, storage falls back to a local `.data/` directory
so you can develop the full loop offline.

## Deploy to Vercel

1. Push to GitHub and **Import** the repo at https://vercel.com/new
   (or run `npx vercel` from this folder).
2. In the project, **Storage → create a Blob store** and connect it. That injects
   `BLOB_READ_WRITE_TOKEN`, which the shared dataset/model storage requires in
   production. Redeploy.
3. Open the deployment, allow camera access, record + contribute + train.

> Heads-up: the global model is shared and unauthenticated — anyone can retrain it.
> Fine for a demo; add auth/moderation before relying on it.
