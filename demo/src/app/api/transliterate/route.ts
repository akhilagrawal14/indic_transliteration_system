import { NextRequest, NextResponse } from "next/server";

// Same-origin proxy: the browser calls /api/transliterate on the frontend, and
// this server-side handler forwards to the backend. The backend URL is a
// server-only env var (BACKEND_URL), never sent to the browser, so the backend
// can stay private (only the frontend is exposed, e.g. via a Cloudflare tunnel).
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const backend = process.env.BACKEND_URL || "http://localhost:8000";
  const qs = req.nextUrl.searchParams.toString();
  try {
    const res = await fetch(`${backend}/transliterate?${qs}`, { cache: "no-store" });
    const body = await res.text();
    return new NextResponse(body, {
      status: res.status,
      headers: { "content-type": "application/json" },
    });
  } catch {
    // Backend unreachable: signal the client to use its offline dictionary.
    return NextResponse.json({ error: "backend unavailable" }, { status: 503 });
  }
}
