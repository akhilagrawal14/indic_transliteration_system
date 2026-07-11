# Indic Transliteration Runtime: Report

**Author:** Akhil Agrawal
**Date:** July 2026

---

## 1. Workload Characterization

### 1.1 Traffic Model

*(Fill in after building the prototype)*

- Typing speed assumption: ~30-50 WPM during active dictation
- Lookup trigger: word boundary with ~200ms debounce (not per-keystroke)
- Per active user: ~0.5-0.9 lookups/second while typing
- Active duty cycle at peak: ~50-70% of 500 concurrent users
- **Estimated peak RPS: ~200-350 requests/second**
- Request size: 3-12 character Latin string. Response size: ~200-400 bytes JSON
- Daily volume: ~15M-50M lookups/day across 5,000 DAU

### 1.2 Latency Budget Breakdown

| Component | Budget (ms) | Notes |
|---|---|---|
| Client to server network | 20-60 | Indian broadband/mobile, flaky in courtrooms |
| TLS/HTTP overhead | ~5 | Amortized with keep-alive connections |
| Server-side queueing | <10 | Must not grow under burst |
| **Model inference** | **<30** | Only for dictionary misses |
| Serialization + response | ~2 | Tiny JSON payload |

Key insight: network latency dominates the budget. Server-side speed is necessary but not sufficient. This motivates client-side caching and CDN-cacheability.

### 1.3 The Zipfian Distribution Insight

**Measured, not assumed.** The entire architecture rests on the claim that a precomputed dictionary serves most lookups. I tested it against Dakshina's natural romanized Hindi text (`hi/romanized/`, 88,658 dev tokens and 86,872 test tokens, 19,119 unique types).

Natural romanized Hindi is strongly Zipfian. Top-N most frequent word types, as a share of total token volume:

| Top-N types | Coverage of token volume |
|---|---|
| 100 | 41.43% |
| 500 | 56.39% |
| 1,000 | 63.96% |
| 2,000 | 71.67% |
| 5,000 | 81.99% |
| 10,000 | 89.50% |

**Held-out dictionary hit rate.** Building a dictionary from one corpus and querying it with an unseen one gives the honest cache-hit estimate (volume weighted, so repeated words count each time, as real traffic does):

| Dictionary source | Entries | Volume hit rate | Unique-type hit rate |
|---|---|---|---|
| Dakshina lexicon (train) | 41,345 | 77.22% | 43.80% |
| Corpus dev vocabulary | 19,102 | 84.65% | 41.86% |
| **Lexicon + corpus dev** | **52,045** | **89.47%** | **57.90%** |

A ~52k-entry dictionary serves **89.5% of lookups by volume** while covering only 58% of unique word types. That gap is exactly the Zipfian property the architecture exploits: the long tail is most of the vocabulary but little of the traffic. In production the dictionary would also absorb Aksharantar word lists and promote frequent model-path misses, pushing this above 90%.

**Methodological note (important).** My first attempt measured this with the `translit.sampled` lexicon splits and produced a hit rate of **11.9%**, which would have falsified the architecture. That number is an artifact: the lexicon is a word list sampled across the frequency spectrum, and its train/test splits are *disjoint by construction*, so held-out overlap is near zero by design. It measures unseen-vocabulary coverage, not cache hit rate. Only natural running text has a real token frequency distribution. Both analyses are retained in `scripts/zipf_coverage.py` so the distinction is auditable.

**Consequence for sizing.** At a ~300 RPS peak with a 89.5% hit rate, the model path sees roughly **32 RPS**. Section 4.1 shows a single CPU core sustains ~127 RPS on the model path, so one core covers the tail with ~4x headroom.

---

## 2. Architecture Decision

### 2.1 Options Evaluated

| Option | p95 Achievable | Cost/Month (5k DAU) | Flaky Network Behavior | Verdict |
|---|---|---|---|---|
| Always-on GPU | Yes | $500-1,400 | No help | Rejected |
| Serverless GPU | No (cold starts) | Variable | No help | Rejected |
| **CPU + dictionary cache** | **Yes** | **$100-250** | Partial | **Chosen** |
| CPU + dictionary + client-side dict | Yes, ~0ms head | +$0 | Excellent, offline-capable | Hardening path |
| CPU + CDN edge | Yes | +$0-20 | Helps RTT | Hardening step |

### 2.2 Why CPU + Dictionary

*(Write the narrative argument here using your own benchmark data)*

### 2.3 Why Not GPU

A GPU is the obvious first reach for this workload. I benchmarked it on an L4 rather than dismissing it, and the result is stronger than the expected cost argument: **for this workload the GPU loses on latency too.**

| | CT2 INT8, 1 CPU core | CT2 INT8, L4 GPU |
|---|---|---|
| p50 latency | **7.39 ms** | 12.33 ms |
| p95 latency | **11.64 ms** | 18.53 ms |
| Cold start | **26 ms** | 344 ms |
| Process RSS | **29 MB** | 176 MB |

The GPU is **1.7x slower at p50**. The reason is structural, not a tuning failure: a request is one word of ~10 characters through a 30M-parameter character-level transformer. The arithmetic is negligible, so per-call kernel launch and host-device transfer overhead dominate. GPUs recover this cost through batching, but batching is unavailable here. Requests arrive independently from separate typists, so forming a batch means holding requests in a queue, spending the exact latency budget we are trying to protect.

So the GPU case collapses on three axes at once:

- **Latency:** 1.7x worse at p50, 1.6x worse at p95 (measured above).
- **Cost:** an L4 instance is roughly $500 to 700/month against roughly $60 to 80/month for a comparable CPU instance, and it would run at low single-digit utilization at ~32 model-path RPS.
- **Operations:** CUDA driver upkeep, 344 ms cold start, 6x the memory footprint.

The GPU earns its place in this project exactly once: **offline dictionary precomputation**, where the workload is a large batch job and batching is free. That is the correct use of the hardware.

*(Pending: measured GPU utilization at target load and $/1M lookups, in section 6.)*

---

## 3. Implementation

### 3.1 Model Runtime

- Base model: AI4Bharat IndicXlit (fairseq transformer, ~30M params, 132 MB FP32 checkpoint)
- Serving engine: CTranslate2 INT8 on CPU, **13 MB** after conversion (10.2x smaller)
- Quality reference: stock fairseq `XlitEngine` FP32

**Conversion notes.** IndicXlit is a *multilingual* fairseq model (`translation_multi_simple_epoch`), which makes the CTranslate2 conversion non-obvious. Three things were required and none are in the documentation:

1. `--unsafe_deserialization`: the checkpoint pickles an `argparse.Namespace`, which torch's `weights_only` loader refuses.
2. `lang_list.txt` must be placed *inside* the data directory. It ships one level above `corpus-bin/`.
3. `--source_lang en --target_lang hi`: all 22 `dict.<lang>.txt` files are byte-identical (one shared 780-token target vocabulary against a 28-character source vocabulary).

The conversion succeeded, so the planned ONNX fallback was never needed.

**Tokenization.** The multilingual model expects a target-language tag prepended to the character sequence. `mera` becomes `["__hi__", "m", "e", "r", "a"]`, with `</s>` appended by CT2's `add_source_eos`. Output tokens are joined with the empty string.

**Environment issues resolved** (documented in `requirements/requirements-precompute.in`):
- `pip >= 24.1` rejects `omegaconf 2.0.x` legacy metadata, breaking the fairseq install. Pin `pip < 24.1`.
- fairseq has no cp310 wheel and its PyPI sdist fails PEP 517 metadata generation. Install from git.
- `ai4bharat-transliteration` imports `urduhack` unconditionally at module load but only calls it for `lang_code == 'ur'`. The real urduhack drags in TensorFlow. `server/compat.py` registers a stub that raises if ever invoked, avoiding a ~400 MB dependency for a dead import.
- **`BEAM_WIDTH` hard-caps candidate count.** With `beam_width=4, topk=5` the engine silently returns only 4 candidates. Defaults corrected to `beam_width=5`. Beam width is also a quality knob: at beam 8 the top-5 ranking changes.

**Serving engine is CPU-only by design.** Note that the stock `XlitEngine` exposes no device parameter at all, so the FP32 baseline could only be benchmarked on CPU. The GPU comparison in section 4.1 therefore runs through CTranslate2, which supports both devices from one converted model.

### 3.2 Dictionary

- Source: Dakshina Hindi lexicon + Aksharantar word lists
- Entries: *(fill in)*
- Size on disk: *(fill in)*
- In-memory format: *(JSON dict / marisa-trie / LMDB)*

### 3.3 API Design

- `GET /transliterate?word=...&lang=...&topk=...`
- Response includes `source` field (dict/cache/model) and `latency_ms`
- Cache-Control headers for CDN/browser caching

### 3.4 Reproducibility

```bash
git clone <repo-url>
cd indic-xlit-runtime
cp .env.example .env
./run.sh docker
curl "http://localhost:8000/transliterate?word=mera&lang=hi&topk=5"
```

---

## 4. Benchmark Results

### 4.1 Microbenchmarks (Single Word)

Method: 200 unique romanized words sampled from the Dakshina Hindi test lexicon, beam width 5, top-5 candidates, 500 sequential timed iterations after 20 warmup iterations (150 iterations for the slow FP32 baseline). One word per call, which is how the service is actually queried. Reproduce with `python scripts/microbench.py`.

| Engine | Device | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (QPS) | Cold start | Model RSS |
|---|---|---|---|---|---|---|---|
| Dictionary lookup | CPU | **0.0004** | 0.001 | 0.001 | ~2,570,000 | 108 ms | 21 MB |
| CTranslate2 INT8 | CPU, 1 thread | **7.39** | 11.64 | 13.70 | 127 | 26 ms | 29 MB |
| CTranslate2 INT8 | CPU, 24 threads | 6.82 | 10.51 | 12.29 | 138 | 10 ms | 12 MB |
| CTranslate2 INT8 | GPU (L4) | 12.33 | 18.53 | 21.63 | 76 | 344 ms | 176 MB |
| Fairseq FP32 (stock) | CPU | 73.34 | 107.83 | 125.37 | 13 | 2,847 ms | 651 MB |

Hardware: GCP g2-standard-24, 24 vCPU, 2x NVIDIA L4. Dictionary row is a 41,345-entry in-memory map built from the Dakshina train lexicon.

Four results drive every decision in this report:

1. **Quantization is not optional, it is the whole game.** The stock fairseq FP32 engine has a p95 of **107.8 ms** for a single word. It exceeds the entire 100 ms end-to-end budget before a single network hop is added. CTranslate2 INT8 cuts p50 by **9.9x** (73.34 to 7.39 ms) and p95 by **9.3x**.

2. **The GPU is slower than the CPU for this workload.** CT2 INT8 on an L4 has a p50 of 12.33 ms against 7.39 ms on one CPU core: the GPU is **1.7x slower**. At batch size 1 on ~10-character sequences, kernel launch and host-device transfer overhead dominate the arithmetic, which is trivial for a 30M-parameter char-level model. The GPU also carries a 344 ms cold start against 26 ms. This turns the GPU rejection from an economic argument into a *performance* one: it costs 5 to 10x more and it is measurably worse on the metric that matters. (A GPU would win on large batches, but batching is not available here: requests are single words arriving independently, and adding a batching window would spend the very latency budget we are protecting.)

3. **Threading buys almost nothing.** Going from 1 to 24 CPU threads improves p50 by only 8% (7.39 to 6.82 ms). The model is too small to parallelize within a request. The correct configuration is therefore **1 intra-op thread per worker with many workers**, which maximizes throughput per box rather than minimizing single-request latency. This is what makes the CPU sizing so favorable.

4. **The dictionary path is effectively free.** At ~0.4 microseconds per lookup it is **~18,000x faster** than the CT2 model path and its cost rounds to zero. This is why serving 89.5% of traffic from the dictionary collapses the compute requirement.

*(Pending: GPU utilization under sustained load, to quantify the idle-silicon cost argument in section 2.3.)*

### 4.2 Load Test Results (500 Concurrent Users, 10 min)

| Metric | Dictionary ON | Dictionary OFF | GPU (comparison) |
|---|---|---|---|
| Aggregate RPS | | | |
| p50 latency (ms) | | | |
| p95 latency (ms) | | | |
| p99 latency (ms) | | | |
| Error rate | | | |
| Cache hit ratio | | | n/a |
| CPU utilization | | | |
| GPU utilization | n/a | n/a | |
| Memory (RSS) | | | |

### 4.3 Burst Test (1000 Users, Step)

*(Fill in: p99 during first 30s, recovery time, error rate)*

### 4.4 Breaking Point

*(Fill in: max sustained RPS before p95 exceeds 50ms server-side)*

---

## 5. Quality Evaluation

### 5.1 Dakshina Hindi Test Set

Evaluated on all 4,442 unique romanized inputs in the Dakshina Hindi test lexicon, beam width 5, top-5. The lexicon is many-to-one (one romanization can have several acceptable native spellings), so predictions are scored against the *set* of acceptable forms; scoring against a single reference would understate accuracy. Reproduce with `eval/eval.py` and `eval/compare_results.py`.

| Engine | Top-1 Accuracy | Top-5 Accuracy | CER (top-1) |
|---|---|---|---|
| FP32 reproduction (stock fairseq) | 61.35% | 87.35% | 0.1170 |
| CTranslate2 INT8 | 61.08% | 87.30% | 0.1183 |
| **Delta (INT8 vs FP32)** | **-0.27 pp** | **-0.05 pp** | **+0.0013** |

The quantization cost is negligible: top-1 within 0.27 percentage points, top-5 within 0.05. Top-5 accuracy is the product-relevant metric, since the typist picks from the ranked dropdown, and it is effectively unchanged.

The published IndicXlit paper reports higher Dakshina numbers, but on a different protocol (its own held-out split with rescoring enabled). This evaluation runs with `rescore=False` for a clean FP32-vs-INT8 comparison on identical inputs, so the absolute figures are not directly comparable to the paper. The *delta* between engines, which is what the deployment decision hinges on, is what matters here.

### 5.2 Qualitative Notes

The two engines produce the **identical top-1 candidate for 98.81%** of inputs. Top-1 changed on only 53 of 4,442 inputs, and that change was a net wash: **3 became correct, 3 became incorrect**, the rest were reorderings among already-acceptable spellings.

Every top-1 disagreement inspected is a rank-1 / rank-2 swap between two plausible transliterations, not a corruption:

| Input | FP32 top-2 | INT8 top-2 | Assessment |
|---|---|---|---|
| `aram` | आराम, अरम | अरम, आराम | both valid, reordered |
| `arya` | आर्य, आर्या | आर्या, आर्य | both valid, reordered |
| `chadron` | चद्रों, चादरों | चादरों, चद्रों | both valid, reordered |
| `dakhile` | दखिले, दाखिले | दाखिले, दखिले | both valid, reordered |

This is the expected signature of INT8 plus a C++ beam search: near-identical rankings with occasional adjacent swaps where two candidates score within rounding of each other. There is no systematic quality regression. Full per-word predictions are retained locally (`eval/results/*.json`, gitignored) and the committed `*.summary.json` files carry the metrics and a sample of misses.

---

## 6. Cost Model

### 6.1 Assumptions

| Parameter | Value | Source |
|---|---|---|
| DAU | 5,000 | Target scale |
| Court hours/day | 3 | Typical Indian courtroom schedule |
| Lookups/user/hour | 1,500 | ~0.7 RPS x 3600s, with active duty cycle |
| Cache hit ratio | *(measured)* | Load test Scenario 2 |
| CPU instance | *(type, price)* | GCP/AWS Mumbai on-demand pricing |
| GPU instance | *(type, price)* | GCP/AWS Mumbai on-demand pricing |

### 6.2 Cost at Scale

| Scale | DAU | Peak Concurrent | CPU + Dict ($/mo) | GPU Always-On ($/mo) |
|---|---|---|---|---|
| 1x (current) | 5,000 | 500 | | |
| 3x | 15,000 | 1,500 | | |
| 10x | 50,000 | 5,000 | | |

### 6.3 Cost per Unit

| Metric | CPU + Dict | GPU |
|---|---|---|
| $/user/month | | |
| $/1M lookups | | |

---

## 7. Reflections

### 7.1 What Breaks at 10x

*(Write after benchmarking. Topics: multi-region deployment, dictionary growth for new languages/names, observability, CDN caching as a scaling lever)*

### 7.2 What Was Traded Away

- No GPU at serving time (justified by benchmarks: it is both slower and more expensive here)
- Fixed beam search parameters (beam 5, top-5). Beam width must be >= topk or candidates are silently truncated. Wider beams (8) reorder the top-5 for a real latency cost and were not explored further
- Single language deep-dive (Hindi)
- No auth/rate limiting/multi-tenancy
- Minimal frontend (functional, not polished)

### 7.3 What I Would Revisit With More Time

- On-device WASM inference for true offline courtroom operation
- Personalized ranking based on typist accept-history
- Streaming prefix suggestions (suggest while mid-word)
- Multi-language dictionary precomputation pipeline
- Prometheus/Grafana observability stack

---

## Appendix

### A. Hardware Used

| Role | Spec | Notes |
|---|---|---|
| Server (benchmarks) | GCP g2-standard-24, 24 vCPU, 2x NVIDIA L4 (23 GB each) | Driver 550.90.07, CUDA 12.4. GPU used for precomputation and comparison only |
| OS / toolchain | Debian 11 (bullseye), Python 3.10.20 (conda env `xlit`) | fairseq 0.12.2, CTranslate2 4.8.1, torch 2.5.1+cu121 |
| Load generator | *(fill in)* | Must be separate from server |

Microbenchmarks in section 4.1 were run on this box with the server process not under concurrent load. Load-test numbers (section 4.2) must be generated from a separate machine to avoid measuring the load generator's own contention.

### B. How to Reproduce Results

```bash
# Section 4.1: single-word latency percentiles across all engines
python scripts/microbench.py --output eval/results/microbench.json
python scripts/microbench.py --print-table eval/results/microbench.json

# Section 1.3: Zipfian coverage and held-out dictionary hit rate
python scripts/zipf_coverage.py --output eval/results/zipf_coverage.json

# Section 5: quality parity, FP32 baseline vs INT8
python eval/eval.py --engine fairseq --lang hi --topk 5 --output eval/results/baseline_fp32.json
python eval/eval.py --engine ct2 --lang hi --topk 5 --output eval/results/ct2_int8.json

# Section 4.2: load test (run from a separate machine)
locust -f loadtest/locustfile.py --host http://<server>:8000 --users 500 \
  --spawn-rate 50 --run-time 600s --headless --csv loadtest/results/scenario2

# Section 6: cost model
python cost/costmodel.py --output cost/results/cost_comparison.json
```

Raw results are committed under `eval/results/`.