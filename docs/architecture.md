# Architecture Decision Records

Documenting the key technical decisions, the alternatives considered, and why each was chosen or rejected. These feed directly into the report.

---

## ADR-1: Serving Engine -- CTranslate2 INT8 on CPU

### Context
IndicXlit is a fairseq transformer seq2seq model (~30M params). We need single-word inference in <30ms on CPU to fit within the overall p95 < 100ms budget after network and queueing overhead.

### Options Considered

| Option | Pros | Cons | Measured Latency (fill in) |
|---|---|---|---|
| Fairseq/PyTorch FP32 on CPU | Zero conversion effort, exact baseline | Python beam search loop is slow, FP32 memory footprint | ~80-150ms (too slow) |
| PyTorch + torch.compile on CPU | Minor speedup, no conversion | Beam search stays in Python, marginal gains | ~60-120ms (still risky) |
| **CTranslate2 INT8 on CPU** | C++ beam search, INT8 quantization, native fairseq converter, production-proven for NMT | Conversion may fail for this specific checkpoint | **~5-25ms (target)** |
| ONNX Runtime INT8 on CPU | Broad hardware support, dynamic quantization | Must split encoder/decoder, write external beam search loop, more work for similar or worse result | ~10-40ms |
| TensorRT on GPU | Fastest raw inference | GPU-only, wrong cost profile for this workload | ~2-5ms (but GPU cost) |

### Decision
CTranslate2 INT8 on CPU, with ONNX Runtime as fallback if conversion fails, and stock fairseq as the quality reference.

### Consequences
- Must validate conversion succeeds for this checkpoint
- Must prove INT8 quality is within ~0.5% of FP32 baseline on Dakshina

---

## ADR-2: Dictionary-First Architecture

### Context
Transliteration is deterministic: `mera` always produces the same ranked candidates. Courtroom vocabulary follows a Zipfian distribution where a small number of common words account for most lookups.

### Options Considered

| Option | Cache Hit Rate | Latency (head) | Complexity | Scales with users? |
|---|---|---|---|---|
| No cache, all model | 0% | 5-25ms every request | Low | Scales linearly with compute |
| **Precomputed dictionary + LRU** | ~90-95% | <1ms for hits | Medium | Sublinear: cache absorbs growth |
| Redis/external cache | ~80-90% (cold start penalty) | 1-3ms (network hop) | Higher (ops burden) | Adds a dependency |
| Client-side dictionary | ~85-90% for head words | 0ms, works offline | Higher (distribution) | Zero server load for hits |

### Decision
Precomputed in-memory dictionary as primary path, in-process LRU cache for recent model results, CTranslate2 model for the tail. Client-side dictionary as a stretch/hardening goal.

### Consequences
- Need to precompute the dictionary offline using the GPU (one-time cost)
- Dictionary size (~100k entries, ~20-50MB) must fit in RAM per worker
- Cache hit ratio is the single most important metric to track in production
- Adding a new language means precomputing a new dictionary

---

## ADR-3: GET Endpoints for Cacheability

### Context
The transliteration API response is purely a function of its input. There is no user state, no session, no side effects.

### Decision
Use `GET /transliterate?word=...&lang=...&topk=...` instead of POST. Set `Cache-Control: public, max-age=86400` since outputs never change.

### Consequences
- Browsers cache responses automatically (free client-side caching)
- CDNs (Cloudflare, etc.) can cache at the edge for free (free geographic distribution)
- Reverse proxies (nginx) can cache upstream responses
- Query string length is not a concern: inputs are single short words
- No CORS preflight overhead (GET is a simple request)

---

## ADR-4: No GPU at Serving Time

### Context
The obvious always-on-GPU answer is expensive, and that cost grows with every courtroom onboarded.

### Analysis

| Metric | CPU + Dict (chosen) | Always-on L4 GPU |
|---|---|---|
| Inference latency (model path) | ~10-20ms | ~3-5ms |
| Dictionary path latency | <1ms | <1ms (same) |
| End-to-end p95 (mixed, 500 users) | ~15-25ms | ~5-10ms |
| GPU utilization | n/a | <5% (massive waste) |
| Monthly cost (single instance) | ~$60-80 | ~$500-700 |
| Monthly cost (2x for HA) | ~$120-160 | ~$1,000-1,400 |
| Cost at 10x users | ~$300-500 | ~$3,000-5,000 |
| Cold start | ~2s (process start) | ~10-30s (GPU init) |
| Ops burden | Standard Linux | CUDA drivers, GPU monitoring |

### Decision
CPU serving. GPU is used only for offline precomputation and benchmarking.

### Consequences
- Must benchmark the GPU path honestly and include the numbers in the report
- The latency difference on the model path (10ms vs 3ms) is invisible to users because (a) 90%+ of requests never hit the model, and (b) the network RTT to courtrooms dwarfs both

---

## ADR-5: Locust with Zipfian Distribution

### Context
Load test realism directly affects the credibility of benchmark numbers. A uniform random word distribution produces artificially low cache hit rates and artificially high model-path load.

### Decision
The Locust load test draws words from the Dakshina vocabulary with frequency weighting (Zipfian). Each simulated "typist" user emits lookups at ~0.7 requests/second (word-boundary debounce, ~40 WPM typing speed, not every word triggers a lookup).

### Consequences
- Cache hit ratios in the load test will match production behavior (~90%+)
- The cache-off ablation scenario (Scenario 4) provides the counterfactual: what the system looks like without the dictionary
- Comparing Scenario 2 (realistic) vs Scenario 4 (all-model) is the strongest evidence for the dictionary architecture

---

## ADR-6: Python 3.10 (Not 3.11+)

### Context
fairseq 0.12.2 and the ai4bharat-transliteration library have known compatibility issues with Python 3.11+ (deprecated imports, NumPy API changes, Cython build failures).

### Decision
Pin Python 3.10 via conda. Do not use 3.11 or 3.12 features.

### Consequences
- Limits access to newer Python features (ExceptionGroup, tomllib, etc.), none of which matter for this project
- Avoids hours of debugging dependency hell, a common time sink on fairseq-based projects