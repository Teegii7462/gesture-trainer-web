import { NextResponse } from "next/server";
import { FEATURE_DIM } from "@/lib/gesture/features";
import { SCHEMA_VERSION, type ModelBundle } from "@/lib/gesture/model";
import { putModel, readModel } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const bundle = await readModel();
  if (!bundle) {
    return NextResponse.json({ error: "no model yet" }, { status: 404 });
  }
  return NextResponse.json(bundle);
}

export async function POST(req: Request) {
  let bundle: ModelBundle;
  try {
    bundle = (await req.json()) as ModelBundle;
  } catch {
    return NextResponse.json({ ok: false, error: "invalid JSON" }, { status: 400 });
  }
  // minimal validation before overwriting the shared model
  if (
    !bundle ||
    bundle.schema !== SCHEMA_VERSION ||
    bundle.config?.featureDim !== FEATURE_DIM ||
    !bundle.forest?.trees?.length
  ) {
    return NextResponse.json({ ok: false, error: "invalid model bundle" }, { status: 400 });
  }
  try {
    await putModel(bundle);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: err instanceof Error ? err.message : "storage error" },
      { status: 500 },
    );
  }
}
