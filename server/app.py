"""FastAPI transliteration service.

Serving path: precomputed dictionary -> in-process LRU cache -> model. All
endpoints are GET so responses are cacheable by browsers, CDNs, and proxies
(transliteration is deterministic per input).

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
from typing import Deque, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.config import get_settings
from server.dictionary import DictionaryCache

# Romanized input is latin letters only. Reject anything else (digits,
# punctuation, other scripts) rather than feeding it to the model.
WORD_RE = re.compile(r"^[A-Za-z]+$")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Indic Transliteration Runtime")

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


@app.on_event("startup")
def _startup() -> None:
    global _engine, _cache
    _cache = DictionaryCache(dict_path=settings.dict_path,
                             lru_size=settings.lru_cache_size)
    _engine = _build_engine()
    logger.info("Ready: engine=%s device=%s dict_size=%d",
                settings.engine, settings.device, _cache.size)


def _record(source: str, elapsed_ms: float) -> None:
    with _metrics_lock:
        _counts["total"] += 1
        _counts[source] += 1
        _latencies.append(elapsed_ms)


@app.get("/transliterate")
def transliterate(
    word: str = Query(..., min_length=1, max_length=64),
    lang: str = Query(default=None),
    topk: int = Query(default=None, ge=1, le=settings.topk),
) -> JSONResponse:
    """Return ranked Indic candidates for a romanized word.

    The response is a pure function of `word` (and the fixed lang/model/beam), so
    it is deterministic and safe to cache. The LRU always stores the canonical
    full candidate list (`settings.topk` entries) regardless of the requested
    `topk`, and the response slices to `topk`; this prevents a small-topk request
    from poisoning the cache for a later larger-topk request. The LRU is
    process-local and cleared on restart, so a model change (which requires a
    restart and dictionary regeneration) cannot serve stale cached values.
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
        # Always compute/store the canonical full list, then slice per request.
        candidates = _engine.transliterate(key, topk=settings.topk)
        _cache.store(key, candidates)
        source = "model"

    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    _record(source, elapsed_ms)

    return JSONResponse(
        {
            "input": word,
            "candidates": candidates[:k],
            "source": source,
            "latency_ms": elapsed_ms,
        },
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/healthz")
def healthz() -> Dict[str, object]:
    """Liveness plus a quick view of what the worker loaded."""
    return {"status": "ok", "engine": settings.engine, "dict_size": _cache.size}


@app.get("/metrics")
def metrics() -> Dict[str, object]:
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

    return {
        "counts": counts,
        "hit_ratio": {
            "dict": round(counts["dict"] / total, 4),
            "cache": round(counts["cache"] / total, 4),
            "model": round(counts["model"] / total, 4),
        },
        "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                       "samples": len(samples)},
    }
