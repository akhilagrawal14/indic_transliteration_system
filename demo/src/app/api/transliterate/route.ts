import { NextRequest, NextResponse } from "next/server";

// Same-origin proxy: the browser calls /api/transliterate on the frontend, and
// this server-side handler forwards to the backend. The backend URL is a
// server-only env var (BACKEND_URL), never sent to the browser, so the backend
// can stay private (only the frontend is exposed, e.g. via a Cloudflare tunnel).
export const dynamic = "force-dynamic";

// Product-level deadline for the backend hop. A word lookup must feel instant, so
// if the backend does not answer within this budget we give up and let the client
// fall back to its bundled offline dictionary. This bounds the failure not just
// when the backend is unreachable (fast connection refused) but also under a
// degraded/half-open connection, where a plain fetch could hang for seconds.
const BACKEND_TIMEOUT_MS = 120;

export async function GET(req: NextRequest) {
  const backend = process.env.BACKEND_URL || "http://localhost:8000";
  const qs = req.nextUrl.searchParams.toString();
  try {
    // Time the backend call from the proxy's side. Returned to the client as
    // `proxy_ms` so the UI can decompose the browser-measured round-trip into
    // backend compute (`latency_ms`), this proxy->backend leg (`proxy_ms`, which
    // is intra-Docker network + backend), and the remainder = browser<->frontend
    // transit (the port-forward tunnel). That remainder is where the gap lives.
    const t0 = performance.now();
    const res = await fetch(`${backend}/transliterate?${qs}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });
    const body = await res.text();
    const proxyMs = Math.round((performance.now() - t0) * 10) / 10;
    // Attach proxy_ms to the JSON. If the body isn't JSON (unexpected), forward
    // it verbatim rather than failing.
    try {
      const data = JSON.parse(body);
      data.proxy_ms = proxyMs;
      return NextResponse.json(data, {
        status: res.status,
        headers: { "Cache-Control": "no-store" },
      });
    } catch {
      return new NextResponse(body, {
        status: res.status,
        headers: { "content-type": "application/json", "Cache-Control": "no-store" },
      });
    }
  } catch {
    // Backend unreachable or too slow (timeout abort): signal the client to use
    // its offline dictionary.
    return NextResponse.json({ error: "backend unavailable" }, { status: 503 });
  }
}
