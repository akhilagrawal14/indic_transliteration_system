# Indic Transliteration Runtime: Report

**Author:** Akhil Agrawal
**Date:** July 2026

---

## 1. Workload Characterization

### 1.1 Traffic Model

This model drives the load tests (§4.2–4.3) and the cost model (§6), both of which
were built and run. There is no real courtroom traffic to sample yet, so these are
the derived parameters the prototype was tested against (capturing real editor
request traces is the next validation step, §7.3):

- Typing speed: ~30-50 WPM during active dictation
- Lookup trigger: word boundary with ~200ms debounce (not per-keystroke)
- Per active user: ~0.5-0.9 lookups/second while typing
- Active duty cycle at peak: ~50-70% of 500 concurrent users
- **Peak RPS: ~200-350 requests/second** (the target load test ran at ~393 RPS, §4.2)
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

Key insight: network latency dominates the budget. Server-side speed is necessary but not sufficient. This motivates keeping repeat lookups off the network entirely — via the browser session LRU and the bundled offline dictionary (see ADR-3, §3.3) rather than relying on external CDN/HTTP caching.

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

The GPU is **1.7x slower at p50**. The reason is structural, not a tuning failure: a request is one word of ~10 characters through an 11.5M-parameter character-level transformer. The arithmetic is negligible, so per-call kernel launch and host-device transfer overhead dominate. GPUs recover this cost through batching, but batching is unavailable here. Requests arrive independently from separate typists, so forming a batch means holding requests in a queue, spending the exact latency budget we are trying to protect.

So the GPU case collapses on three axes at once:

- **Latency:** 1.7x worse at p50, 1.6x worse at p95 (measured above).
- **Cost:** an always-on L4 instance (g2-standard-4) is $518.99/month against $144.79/month for the recommended CPU instance (n2-standard-4), and it would run at low single-digit utilization at ~32 model-path RPS.
- **Operations:** CUDA driver upkeep, 344 ms cold start, 6x the memory footprint.

The GPU is not needed anywhere in this project, including dictionary precomputation: that is a batch job, but CTranslate2 saturates the CPU cores for it and builds the full ~52k-word dictionary in ~100 s (section 3.2), so no GPU is used. The L4 appears only as the *rejected* option we benchmarked to make this decision quantitative.

*(Pending: measured GPU utilization at target load and $/1M lookups, in section 6.)*

---

## 3. Implementation

### 3.1 Model Runtime

- Base model: AI4Bharat IndicXlit (fairseq transformer seq2seq, **~11.5M params** — verified 11,487,748). The 132 MB `.pt` is a *training* checkpoint, not inference weights: ~44 MB of fp32 weights plus ~88 MB of Adam optimizer state (two momentum buffers per parameter), which inference discards.
- Serving engine: CTranslate2 INT8 on CPU, **13 MB** after conversion — ~3.5x smaller than the fp32 weights (~46 MB), ~10x smaller than the full training checkpoint
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
- **Config env-var collision.** A `lang` setting maps to the env var `LANG`, which is the POSIX locale variable (`LANG=C.UTF-8`) and silently overrode the intended `hi`, so model-path requests were tagged with an invalid language. All app settings are now namespaced under an `XLIT_` prefix, which eliminates this and any future collision (e.g. `PORT`).

**Serving engine is CPU-only by design.** Note that the stock `XlitEngine` exposes no device parameter at all, so the FP32 baseline could only be benchmarked on CPU. The GPU comparison in section 4.1 therefore runs through CTranslate2, which supports both devices from one converted model.

### 3.2 Dictionary

- Source: Dakshina Hindi train lexicon romanizations + natural romanized corpus dev vocabulary (the ~52k input set that measured 89.5% held-out coverage in section 1.3)
- Entries: **52,045**
- Size on disk: **6.8 MB** JSON
- In-memory format: JSON loaded into a Python dict (O(1) lookup, ~0.0004 ms) — no trie needed at this size (the dict costs ~21 MB RSS)
- Built **CPU-only in 101 s** (1.94 ms/word batched across cores) via `server/precompute.py`. No GPU: CTranslate2 saturates the cores for the batch job, so the precompute the old plan reserved for the L4 is unnecessary
- Ships uncommitted (gitignored, generated by `precompute.py`); the server loads it at startup
- **Browser offline dictionary:** the top **10,000** head words are bundled into the frontend (`demo/public/client_dict_hi.json`) as the offline fallback. Measured on *held-out* natural test traffic (not the corpus it was selected from), it covers **80.72%** of token volume — the honest offline coverage, below the full server dict's 89.47% but enough to keep the demo usable when the backend is unreachable. Reproduce with `scripts/zipf_coverage.py` (row `client_dict_offline`).

### 3.3 API Design

- `GET /transliterate?word=...&lang=...&topk=...`
- Response includes `source` field (dict/cache/model) and `latency_ms`
- Request coalescing (single-flight) is implemented (`server/singleflight.py`, unit-tested): concurrent misses for the same word share one inference, so a suddenly-trending legal name cannot stampede the model. Coalesced followers are currently attributed to `cache` in `/metrics` (a distinct `coalesced` label would sharpen observability).
- Caching is application-owned, **not** HTTP/CDN-based (see ADR-3, option b): responses are served `Cache-Control: no-store`, and the caching benefit comes from the server LRU plus the browser session LRU + offline dictionary. This avoids external caches holding results a model/dict redeploy would invalidate.

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

Hardware: a 4-vCPU CPU instance for the CPU rows; the GPU row was measured separately on a GPU machine with 1x NVIDIA L4 (tested as the rejected path). Dictionary row is a 41,345-entry in-memory map built from the Dakshina train lexicon.

Four results drive every decision in this report:

1. **Quantization is not optional, it is the whole game.** The stock fairseq FP32 engine has a p95 of **107.8 ms** for a single word. It exceeds the entire 100 ms end-to-end budget before a single network hop is added. CTranslate2 INT8 cuts p50 by **9.9x** (73.34 to 7.39 ms) and p95 by **9.3x**.

2. **The GPU is slower than the CPU for this workload.** CT2 INT8 on an L4 has a p50 of 12.33 ms against 7.39 ms on one CPU core: the GPU is **1.7x slower**. At batch size 1 on ~10-character sequences, kernel launch and host-device transfer overhead dominate the arithmetic, which is trivial for an 11.5M-parameter char-level model. The GPU also carries a 344 ms cold start against 26 ms. This turns the GPU rejection from an economic argument into a *performance* one: it costs 5 to 10x more and it is measurably worse on the metric that matters. (A GPU would win on large batches, but batching is not available here: requests are single words arriving independently, and adding a batching window would spend the very latency budget we are protecting.)

3. **Threading buys almost nothing.** Going from 1 to 24 CPU threads improves p50 by only 8% (7.39 to 6.82 ms). The model is too small to parallelize within a request. The correct configuration is therefore **1 intra-op thread per worker with many workers**, which maximizes throughput per box rather than minimizing single-request latency. This is what makes the CPU sizing so favorable.

4. **The dictionary path is effectively free.** At ~0.4 microseconds per lookup it is **~18,000x faster** than the CT2 model path and its cost rounds to zero. This is why serving 89.5% of traffic from the dictionary collapses the compute requirement.

The GPU path is not carried into the load test: the microbenchmark already shows it is slower here, so there is nothing to gain by driving it under concurrency.

### 4.1b Precision Frontier (why INT8, with data)

To justify INT8 rather than assert it, I swept every CTranslate2 CPU compute type through both the latency microbench and the full quality eval (4,442 Dakshina test inputs, beam 5, 1 CPU thread). Reproduce with `python scripts/frontier.py --sweep precision`.

| compute_type | p50 (ms) | p95 (ms) | Top-1 | Top-5 | CER |
|---|---|---|---|---|---|
| **int8 (chosen)** | **7.5** | 11.8 | 61.08% | **87.30%** | 0.1183 |
| int8_float32 | 7.5 | 11.8 | 61.08% | 87.30% | 0.1183 |
| int16 | 11.3 | 17.4 | 61.23% | 87.30% | 0.1174 |
| float32 | 19.4 | 27.3 | 61.17% | 87.30% | 0.1176 |

INT8 is Pareto-optimal. Three points settle the precision question:

1. **Top-5 accuracy is identical (87.30%) at every precision.** Top-5 is the product metric (the typist picks from the ranked list), so there is literally zero product-relevant quality to recover by spending latency on higher precision.
2. **Higher precision costs 1.5x to 2.6x the latency for ≤0.15 pp of top-1.** int16 is 1.5x slower for +0.15 pp top-1; float32 is 2.6x slower for +0.09 pp. Not a trade worth making.
3. **The INT8 quantization itself is nearly lossless.** CT2 float32 scores 61.17% top-1, so the int8-vs-float32 gap is only 0.09 pp. The remaining ~0.18 pp gap to the fairseq FP32 reference (61.35%, section 5.1) comes from the CTranslate2 conversion and its C++ beam search, not from quantization.

`bfloat16`/`float16` are not in this table because CTranslate2 supports them only on GPU, and the CPU instance used here predates AVX-512-BF16/AMX in any case. They would require a Sapphire Rapids (C3) instance and a different runtime, which was outside the benchmark environment (see section 5.4 and 7.3).

### 4.1c Beam Width Frontier

Beam width is the other decode-time knob. Sweeping it (fixed INT8) shows why 5 is the floor and 8 is not worth it. Reproduce with `python scripts/frontier.py --sweep beam`.

| beam | p50 (ms) | Top-1 | Top-k accuracy | candidates (k) |
|---|---|---|---|---|
| 1 (greedy) | 4.7 | 60.87% | 60.87% | 1 |
| 3 | 6.6 | 60.60% | 82.15% | 3 |
| **5 (chosen)** | 7.5 | 61.08% | **87.30%** | 5 |
| 8 | 8.8 | 61.23% | 87.53% | 5 |

Beam width caps the number of candidates, so beam < 5 cannot serve a top-5 dropdown at all (greedy returns a single spelling). beam=5 is the minimum that fills the ranked list; beam=8 buys only +0.23 pp top-5 for +17% latency. beam=5 is the right operating point.

### 4.1d ONNX Runtime vs CTranslate2 (why CT2 is the engine)

To measure rather than assert the CT2-vs-ONNX choice, I exported the IndicXlit encoder and decoder to ONNX, drove an external beam search over ONNX Runtime (`server/engine/onnx_engine.py`), and benchmarked it against CT2. Latency is single-word (1 CPU thread); quality is a 1,000-word Dakshina test sample (same sample for all three, so the absolute figures differ slightly from the full-set numbers in section 5.1). Reproduce with `python scripts/export_onnx.py` then `python scripts/onnx_compare.py`.

| Engine | p50 (ms) | p95 (ms) | Top-1 | Top-5 | CER |
|---|---|---|---|---|---|
| **CTranslate2 INT8 (chosen)** | **7.6** | 11.8 | 58.9% | 86.7% | 0.128 |
| ONNX Runtime FP32 | 490 | 546 | 59.1% | 87.9% | 0.136 |
| ONNX Runtime INT8 (dynamic) | 283 | 313 | 59.3% | 87.7% | 0.137 |

Two conclusions:

1. **Quality is identical across runtimes.** All three land within sampling noise on this 1,000-word set (top-1 58.9 to 59.3%, top-5 86.7 to 87.9%). A different runtime's INT8 does not change the outputs, which reinforces section 5.3: the quantization is effectively lossless regardless of engine.
2. **CT2 is ~37x faster than ONNX INT8 on CPU** (7.6 ms vs 283 ms p50), and ONNX INT8 is itself ~1.7x faster than ONNX FP32 (283 vs 490 ms), so quantization helps ONNX too but cannot close the gap. CT2 wins because it is purpose-built for transformer NMT decoding: fused C++ kernels and a C++ beam search, versus a generic ONNX graph driven by a Python beam loop.

**Fair-comparison caveat.** The exported ONNX decoder has no KV cache (fairseq's incremental decoding does not export cleanly), so it re-runs the full prefix each step; its latency is therefore an upper bound. A production ONNX path with caching or ORT's fused BeamSearch operator would be several times faster, but it would still carry the Python-orchestration and generic-graph overhead, and would still trail CT2's native fairseq path. The export itself, splitting encoder and decoder and hand-writing the beam search, is precisely the effort CT2's native converter spared us. This is the concrete basis for choosing CT2 and keeping ONNX as a documented fallback only.

### 4.1e Why not vLLM / large-LLM serving stacks

A reasonable question is whether a modern high-throughput serving engine (vLLM, TGI, TensorRT-LLM, etc.) could go faster on CPU. It cannot, for a structural reason: those stacks are built for **large decoder-only autoregressive LLMs**. Their wins — PagedAttention KV-cache management, continuous/in-flight batching, tensor parallelism — all target models where a single request is hundreds-to-thousands of tokens through billions of parameters, and where many such requests can be batched on a GPU.

This workload is the opposite on every axis:

- **Model shape.** IndicXlit is an **11.5M-parameter encoder–decoder (seq2seq) character-level transformer**, not a decoder-only LLM. vLLM and TGI do not serve arbitrary fairseq NMT seq2seq models; adopting one would mean re-implementing the architecture in their supported model set first.
- **Request shape.** One request is a single ~10-character word producing ~10 output characters. The arithmetic is trivial; per-call *overhead* dominates. That is exactly why the L4 GPU is 1.7x **slower** than one CPU core here (§4.1), and why the Python-driven ONNX path is ~37x slower than CT2 (§4.1d). A heavier Python scheduler (vLLM's) would land in the same regime, not CT2's.
- **Batching is unavailable.** Continuous batching — the main vLLM throughput lever — requires a queue of concurrent requests. Ours arrive independently from separate typists, and holding a batching window would spend the very latency budget we are protecting (§2.3).
- **CT2 already is the specialized runtime.** CTranslate2 is purpose-built for transformer-NMT decoding with fused C++ kernels and a native C++ beam search, and measures **7.39 ms p50 on one CPU core** (§4.1). Threading past one intra-op thread buys only ~8% (§4.1, point 3), so there is little single-request headroom left to chase.

The decisive point is architectural, not a benchmark gap: **the dictionary already serves ~89.5% of traffic at ~0.4 µs (§1.3), so the model path is a small minority of requests, and on that path CT2 INT8 is already the fastest tested runtime.** There is no CPU runtime swap worth making; CT2 INT8 is at or near the practical floor for this model and workload.

### 4.2 Load Test Results

Load generated with Locust: 500 simulated typists, Zipfian word distribution over Dakshina's natural romanized test corpus (held out from the dictionary), ~0.7 lookups/sec/user. The server was pinned to **4 cores** (`taskset -c 0-3`, 4 uvicorn workers, 1 CT2 thread each) to approximate an n2-standard-4; this is a CPU-capacity experiment, not a full-instance benchmark (no TLS, load balancer, CDN, network RTT, or GCP scheduler quota). Locust ran on the remaining cores. Percentiles are end-to-end (Locust client side); server-side model latency is from `/metrics`.

| Metric | INT8, dict ON (chosen) | FP32, dict ON | INT8, dict OFF (ablation) |
|---|---|---|---|
| Aggregate RPS | **393** | 215 | 384 |
| p50 (ms) | **2** | 370 | 18 |
| p95 (ms) | **12** | 6,200 | 54 |
| p99 (ms) | **20** | 9,900 | 87 |
| max (ms) | 82 | 19,000 | 220 |
| Error rate | 0% | 0% | 0% |
| Cache hit ratio | 92.0% | 90.5% | 0% (disabled) |
| Requests served | 117,762 | 64,430 | 68,855 |

Three conclusions:

1. **The chosen config clears the bar with enormous margin.** INT8 + dictionary holds p95 = 12 ms end-to-end at 393 RPS on 4 cores, against a target of p95 < 100 ms (and an internal server-side bar of 50 ms). This is the n2-standard-4 doing ~1.3x the realistic peak (~300 RPS) with room to spare, zero errors.

2. **FP32 collapses under load even with the same 90% cache.** p95 balloons to 6.2 seconds. The cause is structural: a fairseq FP32 inference takes ~76 ms and holds the worker (Python-side beam search), so the 9.5% of requests that miss the cache block the workers, and the fast dictionary hits queue behind them. Aggregate throughput falls to 215 RPS because the closed-loop users stall waiting on multi-second responses. This is the load-test proof that quantization is not an optimization but a requirement: single-word latency (section 4.1, p95 108 ms) already exceeded budget, and under concurrency it becomes catastrophic.

3. **INT8 is the load-bearing component; the dictionary is the margin.** With the cache switched off entirely (every request hits the model), INT8 still sustains 384 RPS at p95 = 54 ms (server-side model p95 = 34.6 ms, under the 50 ms bar). So the quantized CPU model alone nearly meets the target; the dictionary then drops the model-path load ~12x (to ~8% of traffic), cutting p50 from 18 ms to 2 ms and, more importantly, freeing headroom so one small box absorbs 10x growth. The ablation delta (p95 12 ms with cache vs 54 ms without) is modest in absolute terms precisely because INT8 is already fast, but it is what turns a "one instance per ~400 RPS" system into a "one instance per several thousand RPS" system.

### 4.3 Burst, Sustained Load, Allocator, and the Ceiling

More INT8 runs on the same 4-core config characterize the operating envelope. The two **Sustained** rows are 5-minute steady-state runs (ramp to a fixed user count, then hold) added to confirm the burst numbers persist over time, not just for a few seconds:

| Run | Offered load | RPS | p50 | p95 | p99 | p99.9 | max | errors |
|---|---|---|---|---|---|---|---|---|
| Target (baseline) | 500 users @ ~0.7 rps | 393 | 2 | 12 | 20 | 49 | 82 | 0% |
| + jemalloc allocator | 500 users @ ~0.7 rps | 393 | 2 | 13 | 21 | 50 | 86 | 0% |
| Burst (step) | 1000 users all at once | 788 | 6 | 48 | 190 | 780 | 870 | 0% |
| **Sustained (5 min)** | **1000 users @ ~0.7 rps** | **782** | **6** | **35** | **66** | **130** | **260** | **0%** |
| **Sustained (5 min)** | **1300 users @ ~0.7 rps** | **906** | **110** | **300** | **400** | **480** | **580** | **0%** |
| Pessimistic (per-keystroke) | 500 users @ ~3.8 rps | 949 | 200 | 270 | 300 | 330 | 460 | 0% |

**Server-side resource use under load** (4 uvicorn workers, sampled mid-run via `/proc`):

| Sustained run | Server CPU (of 400% = 4 cores) | Server RSS (all 4 workers) | Hit ratio dict/cache/model | Server-side model p99 |
|---|---|---|---|---|
| 1000 users, 782 RPS | **~130% (~1.3 cores)** | ~1.8 GB (~0.45 GB/worker) | 89.4% / 3.0% / 7.6% | 13.2 ms |
| 1300 users, 906 RPS | **~108% (~1.1 cores)** | ~1.8 GB | 89.5% / 5.3% / 5.2% | 11.0 ms |

- **Burst tolerance.** Spawning 1000 users instantly (2x target, ~788 RPS) is absorbed with zero errors; p95 stays at 48 ms, under the 100 ms budget, and only the p99/p99.9 briefly spike (190/780 ms) before settling. The system degrades gracefully under a courtroom-session-start thundering herd.
- **Sustained load holds.** Held at 1000 concurrent typists (~782 RPS) for 5 minutes: **p95 35 ms, p99 66 ms, 0 errors** — comfortably inside the 100 ms budget, so the burst 788 RPS figure is confirmed *sustainable*, not a transient. The four workers consumed only **~1.3 of 4 cores** and RSS held at ~1.8 GB, so at nearly 2x the realistic peak the box is about **one-third utilized**.
- **The ceiling we measured is the load generator, not the server.** At 1300 users (~906 RPS) end-to-end p50 rose to 110 ms — but server-side latency stayed at **p99 ≤ 14 ms and CPU at ~1.1 cores**, while single-process Locust logged `CPU usage above 90%`. The added latency is queueing *in the load generator*, not the server. This corrects the earlier reading: the ~950 RPS figure from the pessimistic run is a **single-generator limit**, not the server's true ceiling — the 4-core server was never saturated in any of these runs (it peaks at ~1.3 cores). Establishing the real server ceiling needs a **distributed load generator** (multiple Locust workers or separate load boxes); until then ~800 RPS/box is a conservative *floor* verified as sustained, not a hard ceiling.
- **Allocator (jemalloc) buys nothing here.** `LD_PRELOAD=libjemalloc` reproduces the baseline within noise (p95 13 vs 12 ms, p99 21 vs 20 ms). Honest negative result: at INT8's low allocation churn the allocator is not the bottleneck, so we do not adopt it. Worth knowing it was checked rather than assumed.

**Cache-off: the model path's real ceiling (and proof the dictionary is load-bearing).** Disabling both the dictionary and the LRU so ~99% of requests hit CT2 (single-flight still coalesces concurrent duplicates, hence a few % show as `cache`) moves the bottleneck from the load generator to the **server itself**. Sustained 5-minute cache-off runs on the same 4-core box:

| Cache-off run | RPS | p50 | p95 | p99 | max | Server CPU (of 400%) | Server-side model p99 | errors |
|---|---|---|---|---|---|---|---|---|
| 500 users | 390 | 20 | 55 | 87 | 350 | **~323% (~3.2 cores)** | 69 ms | 0% |
| 800 users | 490 | 270 | 1000 | 1200 | 1600 | **~398% (4 cores, pegged)** | 425 ms | 0% |

At 500 users the pure-model path is still in budget (390 RPS, p95 55 ms — reproducing Scenario 4's 384 RPS / 54 ms in §4.2, now confirmed sustained) but already burns ~3.2 of 4 cores. At 800 users the cores peg at ~398% and end-to-end p95 blows past budget to ~1000 ms — **genuine server saturation** (server-side p95 alone is 361 ms), so the pure-model sustainable ceiling is **~400 RPS/box**. This is the architecture's central result in one contrast: the same 4-core box that saturates at ~400 model-RPS serves **≥782 RPS at only ~1.3 cores** once the dictionary absorbs ~90% of traffic (cached rows above). The dictionary is not an optimization layered on a fast model — it is what turns "one box per ~400 RPS" into "one box per several thousand RPS," which is exactly what flattens the cost curve in §6.

### 4.4 Methodology Notes

- Closed-loop users (each waits for a response before the next), so aggregate RPS is latency-limited: a saturated server yields fewer requests rather than errors, which is why the pessimistic and FP32 runs show high latency at 0% failures. No client-side timeout was set; in production those slow responses would be user-abandoned.
- The server ran on 4 vCPUs; the load generator ran on the remaining cores so it never competes with the server. **However, Locust is single-process (one core for its event loop), and it became the bottleneck above ~800 RPS** (it logs `CPU usage above 90%` in the 1300-user run). Because the server-side CPU and latency stayed low there (§4.3), throughput past that point is generator-limited; a distributed generator is needed to find the server's true ceiling.

---

## 5. Quality Evaluation

### 5.1 Dakshina Hindi Test Set

Evaluated on all 4,442 unique romanized inputs in the Dakshina Hindi test lexicon, beam width 5, top-5. The lexicon is many-to-one (one romanization can have several acceptable native spellings), so predictions are scored against the *set* of acceptable forms; scoring against a single reference would understate accuracy. Reproduce with `eval/eval.py` and `eval/compare_results.py`.

| Engine | Top-1 Accuracy | Top-5 Accuracy | CER (top-1) |
|---|---|---|---|
| FP32 reproduction (stock fairseq) | 61.35% | 87.35% | 0.1170 |
| CTranslate2 INT8 | 61.08% | 87.30% | 0.1183 |
| **Delta (INT8 vs FP32)** | **-0.27 pp** | **-0.05 pp** | **+0.0013** |

The quantization cost is negligible: top-1 within 0.27 percentage points, top-5 within 0.05. Top-5 is a reasonable product proxy, since the typist picks from the ranked dropdown, and it is effectively unchanged; but it is only a proxy. Rank matters (a rank-5 pick costs more interaction than rank-1), so a fuller product evaluation would add MRR, mean accepted rank, and keystrokes-saved from real accept logs.

**Scope caveat.** This measures *benchmark-quality preservation* on the Dakshina Hindi lexicon, i.e. that the CT2 INT8 conversion did not regress the baseline model. It is **not** courtroom-domain validation. A production sign-off needs a domain eval set (judge/lawyer/party/village names, legal terminology, English/Hindi code-mixing, abbreviations and case identifiers, noisy typist romanization, prefixes, digits/punctuation, Unicode/nukta normalization). That set does not exist yet and is the top item in section 7.3. So the 87.3% top-5 figure is "preserves the IndicXlit baseline on Dakshina," not "good enough for courtroom typing."

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

### 5.3 Is More Quantization Tuning Worth It?

Short answer: no, and the frontier in section 4.1b shows why with numbers.

What we already do is post-training quantization: the CTranslate2 conversion applies dynamic INT8 PTQ (INT8 weights, activations quantized on the fly). The measured cost of that is 0.27 pp top-1 and 0.05 pp top-5 versus the fairseq FP32 baseline, and section 4.1b decomposes it: only ~0.09 pp is the INT8 step itself (CT2 float32 is already 61.17% vs int8 61.08%); the rest is the CT2 conversion. Top-5, the metric users actually experience, does not move at all.

The levers to try to recover that fraction, and why each is declined:

- **Higher-precision CPU compute types (int8_float32, int16, float32).** Measured in 4.1b: they recover at most 0.15 pp top-1 and 0.00 pp top-5, at 1.5x to 2.6x the latency. Rejected.
- **Calibration-based static INT8 PTQ.** Using a Dakshina calibration set to fix activation scales could in principle beat dynamic INT8, but the available headroom is ~0.09 pp. The ONNX comparison (section 4.1d) settles it empirically: ONNX INT8 (a different runtime's quantization) matches ONNX FP32 and CT2 INT8 to within sampling noise, so quantization is lossless across engines and calibration has nothing to recover. Not pursued.
- **QAT / distillation / retraining.** Out of scope by assignment (the model is given), and pointless given the headroom.
- **bf16 / fp16.** Not available on this CPU or in CTranslate2's CPU backend; would need a Sapphire Rapids (C3) box and a different runtime, outside the benchmark environment. Noted as future work (section 7.3), not evaluated, so no numbers are claimed.

The honest conclusion is that INT8 sits at the knee of the curve: it is the fastest option and its quality is within measurement noise of full precision on the product metric. There is no quantization tuning that improves the user-facing result.

---

## 6. Cost Model

### 6.1 Assumptions

| Parameter | Value | Source |
|---|---|---|
| DAU | 5,000 | Target scale |
| Court hours/day | 3 | Typical Indian courtroom schedule |
| Lookups/user/hour | 1,500 | ~0.7 RPS x 3600s, with active duty cycle |
| Cache hit ratio | **92%** | Measured in load test Scenario 2 |
| Peak RPS (1x) | ~210 | 10% of DAU concurrent x 0.7 rps x 0.6 duty |
| Model-path RPS/core | 100 | Derated from microbench (135 QPS/core ceiling) |
| CPU instance | n2-standard-4 @ **$144.79/mo** | GCP, 4 vCPU / 16 GB / 30 GB |
| GPU instance | g2-standard-4 @ **$518.99/mo** | GCP, 1x L4 |
| HA floor | 2 instances | Behind an HTTPS load balancer |

Reproduce with `python cost/costmodel.py --hit-ratio 0.92`. Output: `cost/results/cost_comparison.json`.

### 6.2 Cost at Scale

| Scale | DAU | Peak RPS | Model RPS | CPU + Dict ($/mo) | GPU Always-On ($/mo) |
|---|---|---|---|---|---|
| 1x (current) | 5,000 | 210 | 17 | **289.58** (2x n2) | 1,037.98 (2x g2) |
| 3x | 15,000 | 630 | 50 | **289.58** (2x n2) | 1,037.98 (2x g2) |
| 10x | 50,000 | 2,100 | 168 | **434.37** (3x n2) | 1,556.97 (3x g2) |

The CPU line is nearly flat: at 1x and 3x the 2-instance HA floor dominates (peak RPS is well under one box's sustained ~800 RPS in-budget floor, §4.3 — and the server was only ~1/3 utilized there, so this is conservative); only at 10x does aggregate throughput require a third box (2,100 / 800 to 3). Note the binding constraint at scale is **aggregate HTTP throughput per box (measured in section 4.3), not the model path** — the model path is only 168 RPS at 10x. The GPU line scales with load and is **3.6x to 5.4x more expensive** at every point. This flat-vs-linear gap is the core economic argument.

### 6.3 Cost per Unit

| Metric | CPU + Dict (1x / 10x) | GPU (1x / 10x) |
|---|---|---|
| $/1k users/month | $57.92 / $8.69 | $207.60 / $31.14 |
| $/1M lookups | $0.43 / $0.06 | $1.54 / $0.23 |

Both curves fall with scale as fixed cost amortizes, but CPU is ~3.6x cheaper per unit throughout. At 10x, the CPU path serves a courtroom user for a fraction of a cent per day of transliteration.

### 6.4 One-Time Experiment Cost

The GPU path was benchmarked on a GPU machine (1x L4) and rejected; all serving and CPU benchmarks then ran on a CPU instance. The GPU test is a one-time evaluation cost, not recurring; the recurring serving cost is the $289.58/month above. Re-run `cost/costmodel.py` as new measurements land to keep this current.

---

## 7. Reflections

### 7.1 What Breaks at 10x

At 10x (50k DAU, ~2,100 peak RPS), the model path is a non-issue (only ~168 RPS), but aggregate HTTP throughput becomes the sizing constraint: at a measured ~800 RPS in-budget per 4-core box, 2,100 RPS needs 3 instances (~$434/month, section 6.2), still 3.6x cheaper than the GPU path. What actually breaks first, in order:

- **Single-region RTT.** For a pan-India deployment, the network dominates the 100 ms budget (section 1.2). Fix: CDN edge caching (GET responses are deterministic and cacheable) and/or multi-region instances. This is additive, not a rewrite.
- **Dictionary staleness.** New names, legal terms, and neologisms miss the precomputed set. Fix: a nightly job that promotes frequent model-path misses into the dictionary (a data flywheel), plus periodic re-precompute as vocabulary grows.
- **Observability.** Aggregate p95 hides per-courtroom tail latency. Fix: per-region/per-courtroom latency dashboards and cache-hit monitoring (the single most important production metric).
- **The worker-blocking failure mode.** The FP32 collapse (section 4.2) is a general warning: any slow synchronous call on the model path can head-of-line-block a worker. Keep the model path fast (INT8) and consider a separate worker pool or async offload if a heavier model is ever introduced.

The migration path is additive: same API, add CDN, add regions, grow the dictionary. That is the mark of a sound architecture choice.

### 7.2 What Was Traded Away

- No GPU at serving time (justified by benchmarks: it is both slower and more expensive here)
- Fixed beam search parameters (beam 5, top-5). Beam width must be >= topk or candidates are silently truncated. Wider beams (8) reorder the top-5 for a real latency cost and were not explored further
- Single language deep-dive (Hindi)
- No auth/rate limiting/multi-tenancy
- Minimal frontend (functional, not polished)

### 7.3 What I Would Revisit With More Time

Validation and hardening (the gap between "benchmarked prototype" and "production-ready"):

- **Courtroom-domain evaluation set** (highest priority). Curate names, legal terms, code-mixing, abbreviations, noisy romanization, prefixes and normalization edge cases, and report MRR / mean accepted rank / keystrokes-saved, not just Dakshina top-5. Without this the quality claim is benchmark preservation, not product fitness (section 5.1).
- **True server ceiling + real-instance testing.** Sustained 5-minute runs at ~782 and ~906 RPS are now done (§4.3) and confirm ~800 RPS/box is comfortably in-budget with the box only ~1/3 utilized — but they also showed the single-process Locust generator, not the server, is what caps throughput past ~800 RPS. Remaining work: a **distributed load generator** (multiple Locust workers / separate load boxes) to find where the server *itself* saturates, plus a real n2-standard-4 behind TLS + a load balancer with an SLO and headroom (the current numbers are a same-box CPU-capacity approximation).
- **Statistical rigor.** Paired bootstrap / McNemar on the paired predictions instead of the informal "within noise"; the 98.81% identical-top-1 and 3-worse/3-better disagreement already point the same way.
- **Production observability.** Replace the debug-only per-worker `/metrics` with Prometheus histograms (multi-process, per-source labels) and per-courtroom p95 dashboards.
- **Offline hardening beyond the client dictionary.** On-device WASM inference and a service-worker cache for true offline operation; personalized accept-history ranking; streaming prefix suggestions.

Runtime/precision (documented, not run here):

- **bf16 / fp16 on Sapphire Rapids (C3 + AMX).** Not evaluated: CTranslate2 has no CPU bf16 path and the CPU instance used predates AVX-512-BF16/AMX. On a C3, AMX-INT8 could also cut INT8 latency further. No numbers are claimed.
- **Calibration-based static INT8 PTQ.** The one quantization lever with any headroom (~0.09 pp); the ONNX comparison (4.1d) already shows it is lossless across engines, so low expected value.
- Multi-language dictionary precomputation pipeline.

---

## Appendix

### A. Hardware Used

| Role | Spec | Notes |
|---|---|---|
| CPU serving + benchmarks | 4-vCPU Linux CPU instance (n2-standard-4 class), 4 uvicorn workers, 1 CT2 thread each | All serving, precompute, eval, and load tests. Approximates the recommended instance; not a full-instance benchmark |
| GPU comparison (rejected path) | A GPU machine with 1x NVIDIA L4 | Used only to benchmark and reject the GPU serving option (section 4.1) |
| OS / toolchain | Debian 11 (bullseye), Python 3.10.20 (conda env `xlit`) | fairseq 0.12.2, CTranslate2 4.8.1, torch 2.5.1 |
| Load generator | Separate host / dedicated cores | Never competes with the server's 4 cores |

Microbenchmarks (section 4.1) were run with the server not under concurrent load. Load-test numbers (section 4.2) kept the server on 4 vCPUs with the load generator isolated, so the results approximate the recommended instance. No GPU is used in serving, precomputation, or load testing; the GPU appears only as the rejected option that was benchmarked separately.

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

# Section 4.2: load test. Actual runs used same-box core pinning (server on
# cores 0-3, Locust on 4-23); a separate load-generator VM is equivalent.
XLIT_ENGINE=ct2 taskset -c 0-3 uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 4 &
taskset -c 4-23 locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --users 500 \
  --spawn-rate 50 --run-time 300s --headless --csv loadtest/results/scenario2_int8

# Section 6: cost model
python cost/costmodel.py --output cost/results/cost_comparison.json
```

Raw results are committed under `eval/results/`.