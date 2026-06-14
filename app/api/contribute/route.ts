import { NextResponse } from "next/server";
import { FEATURE_DIM } from "@/lib/gesture/features";
import { datasetStats, putContribution } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VALID = new Set(["nothing", "beckon", "shoo"]);
const MAX_WINDOWS = 4000;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "invalid JSON" }, { status: 400 });
  }
  const { features, labels } = (body ?? {}) as {
    features?: number[][];
    labels?: string[];
  };
  if (!Array.isArray(features) || !Array.isArray(labels) || features.length !== labels.length) {
    return NextResponse.json({ ok: false, error: "features/labels mismatch" }, { status: 400 });
  }
  if (features.length === 0) {
    return NextResponse.json({ ok: false, error: "no windows" }, { status: 400 });
  }
  if (features.length > MAX_WINDOWS) {
    return NextResponse.json({ ok: false, error: "too many windows" }, { status: 413 });
  }
  for (let i = 0; i < features.length; i++) {
    if (!Array.isArray(features[i]) || features[i].length !== FEATURE_DIM) {
      return NextResponse.json({ ok: false, error: "bad feature length" }, { status: 400 });
    }
    if (!VALID.has(labels[i])) {
      return NextResponse.json({ ok: false, error: `bad label ${labels[i]}` }, { status: 400 });
    }
  }

  try {
    await putContribution({ features, labels });
    const stats = await datasetStats();
    return NextResponse.json({ ok: true, stats });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: err instanceof Error ? err.message : "storage error" },
      { status: 500 },
    );
  }
}
