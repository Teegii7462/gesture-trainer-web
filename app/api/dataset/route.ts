import { NextResponse } from "next/server";
import { readDataset } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { X, y } = await readDataset();
    return NextResponse.json({ X, y });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "storage error" },
      { status: 500 },
    );
  }
}
