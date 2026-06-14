/*
 * Shared dataset + global model storage.
 *
 * Production: Vercel Blob (set BLOB_READ_WRITE_TOKEN by enabling Blob on the
 * project). Each contribution is an append-only blob under `samples/`; the global
 * model is a single blob at `model/global.json`. Per-class window counts are
 * encoded in the sample pathname so /api/stats can sum them from a cheap list()
 * without fetching every body.
 *
 * Local dev (no token): falls back to a `.data/` directory on disk.
 */
import { promises as fs } from "fs";
import path from "path";
import type { ModelBundle } from "./gesture/model";

const LABELS = ["nothing", "beckon", "shoo"] as const;
type Label = (typeof LABELS)[number];

export interface Contribution {
  features: number[][];
  labels: string[];
}

export interface DatasetStats {
  contributions: number;
  windows: number;
  counts: Record<string, number>;
  hasModel: boolean;
}

const useBlob = (): boolean => !!process.env.BLOB_READ_WRITE_TOKEN;
const MODEL_PATH = "model/global.json";
const DATA_DIR = path.join(process.cwd(), ".data");
const LOCAL_SAMPLES = path.join(DATA_DIR, "samples");
const LOCAL_MODEL = path.join(DATA_DIR, "model.json");

function countByLabel(labels: string[]): Record<Label, number> {
  const c: Record<Label, number> = { nothing: 0, beckon: 0, shoo: 0 };
  for (const l of labels) if (l in c) c[l as Label]++;
  return c;
}

function samplePathname(labels: string[]): string {
  const c = countByLabel(labels);
  const id = (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}${Math.random()}`).slice(0, 8);
  return `samples/${Date.now()}_${c.beckon}-${c.shoo}-${c.nothing}_${id}.json`;
}

function countsFromPathname(pathname: string): Record<Label, number> | null {
  const m = pathname.match(/_(\d+)-(\d+)-(\d+)_/);
  if (!m) return null;
  return { beckon: +m[1], shoo: +m[2], nothing: +m[3] };
}

// --------------------------------------------------------------------------
// Contributions
// --------------------------------------------------------------------------
export async function putContribution(c: Contribution): Promise<void> {
  const body = JSON.stringify(c);
  const pathname = samplePathname(c.labels);
  if (useBlob()) {
    const { put } = await import("@vercel/blob");
    await put(pathname, body, { access: "public", addRandomSuffix: false, contentType: "application/json" });
  } else {
    await fs.mkdir(LOCAL_SAMPLES, { recursive: true });
    await fs.writeFile(path.join(DATA_DIR, pathname), body, "utf8");
  }
}

export async function readDataset(): Promise<{ X: number[][]; y: string[] }> {
  const X: number[][] = [];
  const y: string[] = [];
  const bodies = await readAllSampleBodies();
  for (const raw of bodies) {
    try {
      const c = JSON.parse(raw) as Contribution;
      if (Array.isArray(c.features) && Array.isArray(c.labels)) {
        for (let i = 0; i < c.features.length; i++) {
          X.push(c.features[i]);
          y.push(c.labels[i]);
        }
      }
    } catch {
      /* skip malformed */
    }
  }
  return { X, y };
}

async function readAllSampleBodies(): Promise<string[]> {
  if (useBlob()) {
    const { list } = await import("@vercel/blob");
    const urls: string[] = [];
    let cursor: string | undefined;
    do {
      const res = await list({ prefix: "samples/", cursor, limit: 1000 });
      for (const b of res.blobs) urls.push(b.url);
      cursor = res.cursor;
    } while (cursor);
    const bodies = await Promise.all(
      urls.map((u) => fetch(u, { cache: "no-store" }).then((r) => r.text())),
    );
    return bodies;
  }
  try {
    const files = await fs.readdir(LOCAL_SAMPLES);
    return Promise.all(
      files.filter((f) => f.endsWith(".json")).map((f) => fs.readFile(path.join(LOCAL_SAMPLES, f), "utf8")),
    );
  } catch {
    return [];
  }
}

// --------------------------------------------------------------------------
// Stats (cheap: from pathnames / filenames, no body fetch)
// --------------------------------------------------------------------------
export async function datasetStats(): Promise<DatasetStats> {
  const counts: Record<string, number> = { beckon: 0, shoo: 0, nothing: 0 };
  let contributions = 0;
  let windows = 0;
  const names = await listSampleNames();
  for (const name of names) {
    const c = countsFromPathname(name);
    contributions++;
    if (c) {
      counts.beckon += c.beckon;
      counts.shoo += c.shoo;
      counts.nothing += c.nothing;
      windows += c.beckon + c.shoo + c.nothing;
    }
  }
  return { contributions, windows, counts, hasModel: await hasModel() };
}

async function listSampleNames(): Promise<string[]> {
  if (useBlob()) {
    const { list } = await import("@vercel/blob");
    const names: string[] = [];
    let cursor: string | undefined;
    do {
      const res = await list({ prefix: "samples/", cursor, limit: 1000 });
      for (const b of res.blobs) names.push(b.pathname);
      cursor = res.cursor;
    } while (cursor);
    return names;
  }
  try {
    return (await fs.readdir(LOCAL_SAMPLES)).filter((f) => f.endsWith(".json"));
  } catch {
    return [];
  }
}

// --------------------------------------------------------------------------
// Global model
// --------------------------------------------------------------------------
export async function putModel(bundle: ModelBundle): Promise<void> {
  const body = JSON.stringify(bundle);
  if (useBlob()) {
    const { put } = await import("@vercel/blob");
    await put(MODEL_PATH, body, {
      access: "public",
      addRandomSuffix: false,
      allowOverwrite: true, // the global model is overwritten on every retrain
      contentType: "application/json",
    });
  } else {
    await fs.mkdir(DATA_DIR, { recursive: true });
    await fs.writeFile(LOCAL_MODEL, body, "utf8");
  }
}

export async function readModel(): Promise<ModelBundle | null> {
  if (useBlob()) {
    const { list } = await import("@vercel/blob");
    const res = await list({ prefix: MODEL_PATH, limit: 1 });
    const blob = res.blobs.find((b) => b.pathname === MODEL_PATH);
    if (!blob) return null;
    try {
      return (await fetch(blob.url, { cache: "no-store" }).then((r) => r.json())) as ModelBundle;
    } catch {
      return null;
    }
  }
  try {
    return JSON.parse(await fs.readFile(LOCAL_MODEL, "utf8")) as ModelBundle;
  } catch {
    return null;
  }
}

async function hasModel(): Promise<boolean> {
  if (useBlob()) {
    const { list } = await import("@vercel/blob");
    const res = await list({ prefix: MODEL_PATH, limit: 1 });
    return res.blobs.some((b) => b.pathname === MODEL_PATH);
  }
  try {
    await fs.access(LOCAL_MODEL);
    return true;
  } catch {
    return false;
  }
}
