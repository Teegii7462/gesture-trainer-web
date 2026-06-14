import { NextResponse } from "next/server";
import { datasetStats } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(await datasetStats());
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "storage error" },
      { status: 500 },
    );
  }
}
