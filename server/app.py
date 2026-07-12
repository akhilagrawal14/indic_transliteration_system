"""FastAPI transliteration service.

Serving path: precomputed dictionary -> in-process LRU cache -> model. Endpoints
are GET because a transliteration is a pure, idempotent function of its input
(no user state, no side effects), which keeps the URL the cache key and requests
retry-safe.

Caching policy (see ADR-3): the deployed product does NOT rely on browser/CDN
HTTP caching. The same-origin Next.js proxy fetches with `no-store` and does not
forward upstream cache headers, so responses are not cached at the edge. The
caching benefit comes instead from two application-owned layers: the server's
in-process LRU (model results) and the browser's session LRU + bundled offline
dictionary. Backend responses are therefore served `Cache-Control: no-store` to
avoid an external cache holding results that a model/dictionary redeploy would
invalidate (public caching would require versioned cache keys we do not emit).

Note on metrics: with multiple uvicorn workers each process keeps its own
counters, so /metrics reflects one worker. Traffic is balanced across identical
workers, so the hit ratio is representative; latency percentiles for the report
come from Locust, which sees all workers.
"""

import logging
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from server.config import get_settings
from server.dictionary import DictionaryCache
from server.singleflight import SingleFlight

# Romanized input is latin letters only. Reject anything else (digits,
# punctuation, other scripts) rather than feeding it to the model.
WORD_RE = re.compile(r"^[A-Za-z]+$")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the dictionary/cache and engine once, on startup.

    Replaces the deprecated `@app.on_event("startup")` handler. `_build_engine`
    is looked up on the module at call time, so tests can monkeypatch it with a
    stub engine before entering the TestClient context.
    """
    global _engine, _cache
    _cache = DictionaryCache(dict_path=settings.dict_path,
                             lru_size=settings.lru_cache_size)
    _engine = _build_engine()
    logger.info("Ready: engine=%s device=%s dict_size=%d",
                settings.engine, settings.device, _cache.size)
    yield


app = FastAPI(title="Indic Transliteration Runtime", lifespan=lifespan)

# Allow the browser demo (a different origin, e.g. localhost:3000) to call the
# API. Responses are public and GET-only, so a permissive default is fine;
# narrow XLIT_CORS_ORIGINS in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Built at startup.
_engine = None
_cache: DictionaryCache = None  # type: ignore[assignment]

# Coalesce concurrent model misses for the same word into one inference.
_singleflight = SingleFlight()

# Lightweight in-process metrics.
_counts: Dict[str, int] = {"total": 0, "dict": 0, "cache": 0, "model": 0}
_latencies: Deque[float] = deque(maxlen=5000)
_metrics_lock = threading.Lock()


def _build_engine():
    """Construct the configured engine. Imported lazily to keep startup cheap."""
    if settings.engine == "ct2":
        from server.engine.ct2_engine import CT2Engine
        return CT2Engine(
            settings.model_dir, lang=settings.lang, beam_width=settings.beam_width,
            topk=settings.topk, device=settings.device, intra_threads=1,
        )
    if settings.engine == "fairseq":
        from server.engine.fairseq_engine import FairseqEngine
        return FairseqEngine(
            lang=settings.lang, beam_width=settings.beam_width, topk=settings.topk,
        )
    raise ValueError(f"unknown engine: {settings.engine}")


def _record(source: str, elapsed_ms: float) -> None:
    with _metrics_lock:
        _counts["total"] += 1
        _counts[source] += 1
        _latencies.append(elapsed_ms)


class TransliterationResponse(BaseModel):
    input: str
    candidates: list[str]
    source: str
    latency_ms: float


class LiveResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    engine: str
    dict_size: int


class HitRatio(BaseModel):
    # `dict_` avoids shadowing BaseModel.dict; serialized as "dict" via alias so the
    # JSON shape is unchanged.
    model_config = ConfigDict(populate_by_name=True)
    dict_: float = Field(alias="dict")
    cache: float
    model: float


class LatencyStats(BaseModel):
    p50: float
    p95: float
    p99: float
    samples: int


class MetricsResponse(BaseModel):
    counts: Dict[str, int]
    hit_ratio: HitRatio
    latency_ms: LatencyStats


@app.get("/transliterate", response_model=TransliterationResponse)
def transliterate(
    response: Response,
    word: str = Query(..., min_length=1, max_length=64),
    lang: str = Query(default=None),
    topk: int = Query(default=None, ge=1, le=settings.topk),
) -> TransliterationResponse:
    """Return ranked Indic candidates for a romanized word.

    API contract: `word` is a single *normalized token* — latin letters only,
    no whitespace, punctuation, digits, or apostrophes/hyphens. The frontend is
    responsible for extracting clean tokens (it does; see `wordAtCursor` in the
    demo). Anything else is rejected 422 rather than being fed to the model; this
    endpoint is deliberately not a general "romanized input" cleaner.

    The response is a pure function of `word` (and the fixed lang/model/beam), so
    it is deterministic. The LRU always stores the canonical full candidate list
    (`settings.topk` entries) regardless of the requested `topk`, and the response
    slices to `topk`; this prevents a small-topk request from poisoning the cache
    for a later larger-topk request. The LRU is process-local and cleared on
    restart, so a model change (which requires a restart and dictionary
    regeneration) cannot serve stale cached values. The response is `no-store`
    (see module docstring / ADR-3): correctness relies on the process-local LRU
    and the client's session LRU, not on any external HTTP cache that a redeploy
    could leave stale.
    """
    if lang is not None and lang != settings.lang:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported lang '{lang}'; this deployment serves "
                   f"'{settings.lang}'",
        )
    if not WORD_RE.match(word):
        raise HTTPException(status_code=422,
                            detail="word must contain only latin letters")

    start = time.perf_counter()
    k = topk if topk is not None else settings.topk
    key = word.lower()

    candidates, source = _cache.lookup(key)
    if candidates is None:
        # Miss: coalesce concurrent misses for the same key so only one thread
        # runs inference. The leader computes/stores the canonical full list;
        # followers wait and reuse it, and are attributed to the cache (they did
        # not run the model) so metrics reflect true inference count.
        def _compute() -> list:
            cands = _engine.transliterate(key, topk=settings.topk)
            _cache.store(key, cands)
            return cands

        candidates, was_leader = _singleflight.do(key, _compute)
        source = "model" if was_leader else "cache"

    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    _record(source, elapsed_ms)

    # Correctness relies on the process-local + client LRUs, not external caches.
    response.headers["Cache-Control"] = "no-store"
    return TransliterationResponse(
        input=word, candidates=candidates[:k], source=source, latency_ms=elapsed_ms,
    )


@app.get("/livez", response_model=LiveResponse)
def livez() -> LiveResponse:
    """Liveness: the process and event loop are up. Does not touch the model or
    dictionary, so it stays ok during startup and never restarts a healthy worker
    that is merely still loading. Use this for the orchestrator's liveness probe."""
    return LiveResponse(status="ok")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Readiness: ok only once the dictionary and engine have finished loading.
    Returns 503 during startup so a load balancer holds traffic until the worker
    can actually serve. Use this for the readiness probe."""
    if _cache is None or _engine is None:
        raise HTTPException(status_code=503, detail="starting up")
    return HealthResponse(status="ok", engine=settings.engine, dict_size=_cache.size)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Request counts, hit ratios, and running latency percentiles (one worker)."""
    with _metrics_lock:
        counts = dict(_counts)
        samples = sorted(_latencies)

    total = counts["total"] or 1

    def pct(p: float) -> float:
        if not samples:
            return 0.0
        idx = min(len(samples) - 1, int(p / 100.0 * len(samples)))
        return round(samples[idx], 3)

    return MetricsResponse(
        counts=counts,
        hit_ratio=HitRatio(
            dict_=round(counts["dict"] / total, 4),
            cache=round(counts["cache"] / total, 4),
            model=round(counts["model"] / total, 4),
        ),
        latency_ms=LatencyStats(
            p50=pct(50), p95=pct(95), p99=pct(99), samples=len(samples),
        ),
    )
