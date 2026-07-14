# Indic Transliteration Runtime

**Assignment:** Adalat.ai ML Engineering Take-Home. Build and deploy an Indic transliteration system serving live spelling suggestions to courtroom typists.
**Author:** Akhil Agrawal
**Date:** July 2026

**Code:** https://github.com/akhilagrawal14/indic_transliteration_system/tree/master

**Live demo:** https://logs-coaching-hills-hwy.trycloudflare.com/ (temporary tunnel, valid for at least a week; ping me if it goes down and I will share a fresh link)

---

## Executive Summary

The requirement reads like a GPU problem: p95 under 100ms for 500 concurrent typists. Measured end to end, it is not one.

- **A single 4-core CPU box serves the target load at p95 = 12ms end to end** (393 RPS, zero errors), 8x inside the 100ms budget. The same box sustains ~800 RPS while using only ~1.3 of its 4 cores.
- **The GPU is not just more expensive here, it is slower.** CTranslate2 INT8 on one CPU core: p50 7.39ms. The same engine on an L4 GPU: p50 12.33ms, 1.7x worse. A FLOP budget makes the reason exact: a request is ~0.77 GFLOPs, which an L4 could compute in microseconds, so over 99% of the GPU's 12.33ms is kernel-launch and host-device overhead, not arithmetic. Batching would amortize it, but batching is unavailable because requests arrive independently from separate typists.
- **Two decisions carry the system.** INT8 quantization via CTranslate2 cuts single-word latency 9.9x versus the stock fairseq FP32 engine (73.3ms to 7.4ms p50) and is what makes CPU serving viable at all. A precomputed 52k-entry dictionary then serves a measured 89.5% of held-out traffic at ~0.4 microseconds, collapsing the model path to ~8-11% of requests and flattening the cost curve.
- **Quality is preserved.** INT8 versus the FP32 baseline on the full Dakshina Hindi test lexicon: top-5 accuracy within 0.05 percentage points (87.30% vs 87.35%), identical top-1 candidate on 98.81% of inputs.
- **Cost: $289.58/month at target scale** (2x n2-standard-4 for HA), flat to 3x users, $434/month at 10x. The always-on GPU alternative is 3.6x to 5.4x more expensive at every scale point and never wins on any axis.

The rest of this report shows the measurements behind each of these claims and the alternatives that were benchmarked and rejected, including the GPU, ONNX Runtime, higher precisions, wider beams, and LLM serving stacks.

---

## 1. Workload Characterization

### 1.1 Traffic Model

There is no real courtroom traffic to sample yet, so these are derived parameters, built from typing behavior rather than assumed QPS. They drive the load tests (section 4.3) and the cost model (section 6). Capturing real editor request traces is the first validation step once deployed (section 7.2).

- Typing speed: ~30-50 WPM during active dictation
- Lookup trigger: word boundary with ~200ms debounce, not per keystroke (per-keystroke would multiply load 4-6x for no UX gain; the pessimistic case is still load-tested in section 4.4)
- Per active user: ~0.5-0.9 lookups/second while typing
- Active duty cycle at peak: ~50-70% of 500 concurrent users
- **Peak RPS: ~200-350 requests/second** (the target load test ran at ~393 RPS, above this range)
- Request: a 3-12 character Latin string. Response: ~200-400 bytes of JSON (4-8 ranked candidates)
- Daily volume: ~15M-50M lookups/day across 5,000 DAU

### 1.2 Latency Budget

What p95 < 100ms actually leaves for the server:

| Component | Budget (ms) | Notes |
|---|---|---|
| Client to server network | 20-60 | Indian broadband/mobile, flaky in courtrooms |
| TLS/HTTP overhead | ~5 | Amortized with keep-alive connections |
| Server-side queueing | <10 | Must not grow under burst |
| **Model inference** | **<30** | Only for dictionary misses |
| Serialization + response | ~2 | Tiny JSON payload |

Two conclusions fall out. First, network latency dominates the budget, so server-side speed is necessary but not sufficient; the design keeps repeat lookups off the network entirely via a browser session LRU and a bundled offline dictionary (section 3.3), rather than relying on external CDN/HTTP caching. Second, inference must land in tens of milliseconds, which for an 11.5M-parameter character-level model is CPU territory after quantization.

### 1.3 The Zipfian Distribution Insight (Measured, Not Assumed)

The entire architecture rests on one claim: a precomputed dictionary can serve most lookups. I tested it against Dakshina's natural romanized Hindi text (`hi/romanized/`, 88,658 dev tokens and 86,872 test tokens, 19,119 unique types).

Natural romanized Hindi is strongly Zipfian. Top-N most frequent word types as a share of total token volume:

| Top-N types | Coverage of token volume |
|---|---|
| 100 | 41.43% |
| 500 | 56.39% |
| 1,000 | 63.96% |
| 2,000 | 71.67% |
| 5,000 | 81.99% |
| 10,000 | 89.50% |

**Held-out dictionary hit rate.** Building a dictionary from one corpus and querying it with an unseen one gives the honest cache-hit estimate. Volume weighted, so repeated words count each time, as real traffic does:

| Dictionary source | Entries | Volume hit rate | Unique-type hit rate |
|---|---|---|---|
| Dakshina lexicon (train) | 41,345 | 77.22% | 43.80% |
| Corpus dev vocabulary | 19,102 | 84.65% | 41.86% |
| **Lexicon + corpus dev** | **52,045** | **89.47%** | **57.90%** |

A ~52k-entry dictionary serves **89.5% of lookups by volume** while covering only 58% of unique word types. That gap is exactly the Zipfian property the architecture exploits: the long tail is most of the vocabulary but little of the traffic. In production the dictionary would also absorb Aksharantar word lists and promote frequent model-path misses, pushing this above 90%.

**Methodological note.** My first attempt measured this with the `translit.sampled` lexicon splits and produced a hit rate of **11.9%**, which would have falsified the architecture. That number is an artifact: the lexicon is a word list sampled across the frequency spectrum, and its train/test splits are disjoint by construction, so held-out overlap is near zero by design. It measures unseen-vocabulary coverage, not cache hit rate. Only natural running text has a real token frequency distribution. Both analyses are retained in `scripts/zipf_coverage.py` so the distinction is auditable.

**Consequence for sizing.** At a ~300 RPS peak with an 89.5% hit rate, the model path sees roughly **32 RPS**. Section 4.1 shows a single CPU core sustains ~127 RPS on the model path, so one core covers the tail with ~4x headroom.

---

## 2. Where and How to Run It

### 2.1 Options Evaluated

| Option | p95 achievable | Cost/month (5k DAU) | Flaky-network behavior | Verdict |
|---|---|---|---|---|
| Always-on GPU | Yes | $1,038 (2x g2-standard-4, HA) | No help | Rejected: slower AND costlier here (section 2.3) |
| Serverless GPU | No | Variable, spiky | No help | Rejected: 10-60s cold starts kill an interactive typing UX; keeping instances warm reconverges to always-on pricing |
| **CPU + dictionary cache** | **Yes, measured p95 12ms** | **$290 (2x n2-standard-4, HA)** | Partial (server cache) | **Chosen** |
| + client-side offline dictionary | Yes, ~0ms for head words | +$0 (static asset) | Excellent, offline-capable | Shipped in the demo (section 3.2) |
| + CDN edge caching | Yes | +$0-20 | Helps RTT | Considered and deliberately not used (section 3.3) |

### 2.2 Why CPU + Dictionary

Three measured facts make the choice, not cost intuition alone:

1. **The latency budget is met on CPU with margin.** CT2 INT8 on a single core is p50 7.39ms / p95 11.64ms (section 4.1) against a p95 < 100ms bar. Even with the dictionary switched off entirely, so every request hits the model, the quantized CPU path sustains 390 RPS at p95 55ms (section 4.4). The quantized model alone already clears the budget; GPU-class latency buys nothing the requirement needs, and section 2.3 shows the GPU is actually slower here anyway.

2. **The dictionary collapses the compute, so capacity and cost scale sublinearly.** Transliteration is deterministic per input and natural text is Zipfian, so a precomputed ~52k head dictionary serves ~89.5% of lookups at ~0.4 microseconds (section 1.3), shrinking the model path to ~8-11% of traffic. Measured, this is decisive: the same 4-core box that saturates at ~400 model-RPS with the cache off serves 782+ RPS at only ~1.3 cores with it on (section 4.4). That is what keeps one small box ahead of thousands of RPS and holds cost flat to ~3x scale (section 6).

3. **Operational simplicity.** No CUDA drivers, a 26ms cold start, 29MB RSS. The deployable artifact is a 13MB INT8 model plus a JSON dictionary that loads in ~100ms. Fewer moving parts than any GPU or external-cache option in the table above.

The dictionary is the load-bearing decision; INT8 on CPU is what makes the remaining tail cheap and fast enough. The combination is both faster and ~3.6x cheaper per lookup than the always-on GPU alternative (section 6.3).

### 2.3 Why Not a GPU?

A GPU is the obvious first reach for this workload. I benchmarked it on an L4 rather than dismissing it, and the result is stronger than the expected cost argument: **for this workload the GPU loses on latency too.**

| | CT2 INT8, 1 CPU core | CT2 INT8, L4 GPU |
|---|---|---|
| p50 latency | **7.39 ms** | 12.33 ms |
| p95 latency | **11.64 ms** | 18.53 ms |
| Cold start | **26 ms** | 344 ms |
| Process RSS | **29 MB** | 176 MB |

The GPU is **1.7x slower at p50**. The reason is structural, not a tuning failure: a request is one word of ~10 characters through an 11.5M-parameter character-level transformer. The arithmetic is negligible, so per-call kernel launch and host-device transfer overhead dominate. GPUs recover this cost through batching, but batching is unavailable here: requests arrive independently from separate typists, so forming a batch means holding requests in a queue, spending the exact latency budget we are trying to protect.

The GPU case therefore collapses on three axes at once:

- **Latency:** 1.7x worse at p50, 1.6x worse at p95 (measured above)
- **Cost:** an always-on L4 instance (g2-standard-4) is $518.99/month against $144.79/month for the recommended CPU instance (n2-standard-4), and it would idle at low single-digit utilization at ~32 model-path RPS
- **Operations:** CUDA driver upkeep, a 344ms cold start, 6x the memory footprint

**The FLOP budget: putting a number on "overhead dominates."** From the checkpoint architecture (6-layer encoder + 6-layer decoder, d_model 256, FFN 1024, 4 attention heads, source vocab 54, target vocab 806, 11,487,748 params), a typical lookup costs the following. Assumptions: source S = 12 tokens (~10 chars + `__hi__` tag + `</s>`), T = 11 decode steps, beam 5, with KV caching (cross-attention K/V projected once from the encoder output and cached, not recomputed per step). Convention: FLOPs = 2 x MACs (multiply + add).

| Component | MACs | FLOPs |
|---|---|---|
| Encoder (12 tokens x 6 layers) | 57.1 M | 114 MFLOPs (15%) |
| Decoder (11 steps x 5 beams x 6 layers) | 326.6 M | 653 MFLOPs (85%) |
| **Total per request** | **383.7 M** | **~767 MFLOPs (0.77 GFLOPs)** |

Greedy decoding (beam 1) would be 260 MFLOPs; beam 5 is what makes the decoder dominate. Cross-check: the standard 2 x params x tokens estimate gives ~1.5 GFLOPs, the same ballpark, and the direct count is lower precisely because KV caching avoids re-running the prefix each step.

This budget quantifies both sides of the CPU-vs-GPU result:

- **The CPU is genuinely well utilized.** At the measured p50 of 7.39ms on one core, effective throughput is ~104 GFLOP/s, a healthy fraction of a single core's AVX-512 VNNI INT8 peak. There is no large CPU inefficiency left to reclaim, which is consistent with threading buying only 8% (section 4.1).
- **The GPU spends over 99% of its time on overhead, not math.** On an L4 (~121 TOPS dense INT8), 0.77 GFLOPs of arithmetic would take ~6 microseconds at peak, or ~634 microseconds even at 1% of peak. The measured L4 p50 is 12.33ms. The structural reason is now precise: decoding is 11 sequential steps x 6 layers, each a tiny GEMM (256 x 256), so batch-1 inference is hundreds of small, strictly sequential kernel launches that cannot be overlapped or amortized.

Reproduce with `python scripts/flops.py` (counts derived from the checkpoint config).

The GPU turned out to be unnecessary everywhere in this project, including dictionary precomputation: that is a batch job, but CTranslate2 saturates the CPU cores for it and builds the full ~52k-word dictionary in ~101 seconds (section 3.2). The L4 appears in this report only as the rejected option that was benchmarked to make the rejection quantitative. Per section 6.3 the CPU path is ~3.6x cheaper per 1M lookups at every scale ($0.43 vs $1.54 at 1x, falling to $0.06 vs $0.23 at 10x).

### 2.4 Why Not ONNX Runtime?

To measure rather than assert the engine choice, I exported the IndicXlit encoder and decoder to ONNX, drove an external beam search over ONNX Runtime (`server/engine/onnx_engine.py`), and benchmarked it against CTranslate2. Latency is single-word on 1 CPU thread; quality is a 1,000-word Dakshina test sample (same sample for all three rows, so absolute figures differ slightly from the full-set numbers in section 5.1). Reproduce with `python scripts/export_onnx.py` then `python scripts/onnx_compare.py`.

| Engine | p50 (ms) | p95 (ms) | Top-1 | Top-5 | CER |
|---|---|---|---|---|---|
| **CTranslate2 INT8 (chosen)** | **7.6** | 11.8 | 58.9% | 86.7% | 0.128 |
| ONNX Runtime FP32 | 490 | 546 | 59.1% | 87.9% | 0.136 |
| ONNX Runtime INT8 (dynamic) | 283 | 313 | 59.3% | 87.7% | 0.137 |

Two conclusions:

1. **Quality is identical across runtimes.** All three land within sampling noise on this set (top-1 58.9 to 59.3%, top-5 86.7 to 87.9%). A different runtime's INT8 does not change the outputs, which reinforces section 5.3: the quantization is effectively lossless regardless of engine.
2. **CT2 is ~37x faster than ONNX INT8 on CPU** (7.6ms vs 283ms p50). ONNX INT8 is itself ~1.7x faster than ONNX FP32 (283 vs 490ms), so quantization helps ONNX too, but it cannot close the gap. CT2 wins because it is purpose-built for transformer NMT decoding: fused C++ kernels and a C++ beam search, versus a generic ONNX graph driven by a Python beam loop.

**Fair-comparison caveat.** The exported ONNX decoder has no KV cache (fairseq's incremental decoding does not export cleanly), so it re-runs the full prefix each step; its latency is an upper bound. A production ONNX path with caching or ORT's fused BeamSearch operator would be several times faster, but it would still carry the Python-orchestration and generic-graph overhead, and would still trail CT2's native fairseq path. The export itself, splitting encoder and decoder and hand-writing the beam search, is precisely the effort CT2's native converter spared. This is the concrete basis for choosing CT2 and keeping ONNX as a documented fallback only.

### 2.5 Why Not vLLM / LLM Serving Stacks?

A reasonable question is whether a modern high-throughput serving engine (vLLM, TGI, TensorRT-LLM) could go faster. It cannot, for a structural reason: those stacks are built for large decoder-only autoregressive LLMs. Their wins (PagedAttention KV-cache management, continuous batching, tensor parallelism) all target models where a single request is hundreds to thousands of tokens through billions of parameters, and where many such requests can be batched on a GPU.

This workload is the opposite on every axis:

- **Model shape.** IndicXlit is an 11.5M-parameter encoder-decoder (seq2seq) character-level transformer, not a decoder-only LLM. vLLM and TGI do not serve arbitrary fairseq NMT models; adopting one would mean re-implementing the architecture in their supported model set first.
- **Request shape.** One request is a single ~10-character word producing ~10 output characters. The arithmetic is trivial; per-call overhead dominates. That is exactly why the L4 GPU is 1.7x slower than one CPU core (section 4.1) and why the Python-driven ONNX path is ~37x slower than CT2 (section 2.4). A heavier Python scheduler would land in the same regime, not CT2's.
- **Batching is unavailable.** Continuous batching, the main vLLM throughput lever, requires a queue of concurrent requests. Ours arrive independently from separate typists, and holding a batching window would spend the latency budget we are protecting.
- **CT2 already is the specialized runtime for this model class,** with fused C++ kernels and a native C++ beam search, measuring 7.39ms p50 on one CPU core. Threading past one intra-op thread buys only ~8% (section 4.1), so there is little single-request headroom left to chase.

The decisive point is architectural: the dictionary already serves ~89.5% of traffic at ~0.4 microseconds, so the model path is a small minority of requests, and on that path CT2 INT8 is already the fastest tested runtime. There is no runtime swap worth making.

---

## 3. Implementation

### 3.1 Model Runtime

- Base model: AI4Bharat IndicXlit (fairseq transformer seq2seq, ~11.5M params, verified 11,487,748). The 132MB `.pt` is a training checkpoint, not inference weights: ~44MB of fp32 weights plus ~88MB of Adam optimizer state (two momentum buffers per parameter), which inference discards
- Serving engine: CTranslate2 INT8 on CPU, **13MB after conversion**, ~3.5x smaller than the fp32 weights (~46MB) and ~10x smaller than the full training checkpoint
- Quality reference: stock fairseq `XlitEngine` FP32

**Conversion notes.** IndicXlit is a multilingual fairseq model (`translation_multi_simple_epoch`), which makes the CTranslate2 conversion non-obvious. Three things were required and none are in the documentation:

1. `--unsafe_deserialization`: the checkpoint pickles an `argparse.Namespace`, which torch's `weights_only` loader refuses
2. `lang_list.txt` must be placed inside the data directory; it ships one level above `corpus-bin/`
3. `--source_lang en --target_lang hi`: all 22 `dict.<lang>.txt` files are byte-identical (one shared 780-token target vocabulary against a 28-character source vocabulary)

The conversion succeeded, so the planned ONNX fallback was never needed for serving (it was still built for the engine comparison in section 2.4).

**Tokenization.** The multilingual model expects a target-language tag prepended to the character sequence: `mera` becomes `["__hi__", "m", "e", "r", "a"]`, with `</s>` appended by CT2's `add_source_eos`. Output tokens are joined with the empty string.

**Environment issues resolved** (documented in `requirements/requirements-precompute.in`):

- `pip >= 24.1` rejects `omegaconf 2.0.x` legacy metadata, breaking the fairseq install. Pin `pip < 24.1`
- fairseq has no cp310 wheel and its PyPI sdist fails PEP 517 metadata generation. Install from git
- `ai4bharat-transliteration` imports `urduhack` unconditionally at module load but only calls it for `lang_code == 'ur'`. The real urduhack drags in TensorFlow. `server/compat.py` registers a stub that raises if ever invoked, avoiding a ~400MB dependency for a dead import
- **`BEAM_WIDTH` hard-caps candidate count.** With `beam_width=4, topk=5` the engine silently returns only 4 candidates. Defaults corrected to `beam_width=5`. Beam width is also a quality knob: at beam 8 the top-5 ranking changes (section 4.2)
- **Config env-var collision.** A `lang` setting maps to the env var `LANG`, which is the POSIX locale variable (`LANG=C.UTF-8`) and silently overrode the intended `hi`, tagging model-path requests with an invalid language. All app settings are now namespaced under an `XLIT_` prefix

The stock `XlitEngine` exposes no device parameter at all, so the FP32 baseline could only be benchmarked on CPU. The GPU comparison therefore runs through CTranslate2, which supports both devices from one converted model.

### 3.2 Dictionary

- Source: Dakshina Hindi train lexicon romanizations + natural romanized corpus dev vocabulary (the ~52k input set that measured 89.5% held-out coverage in section 1.3)
- Entries: **52,045**. Size on disk: **6.8MB** JSON
- In-memory format: JSON loaded into a Python dict (O(1) lookup, ~0.0004ms). No trie needed at this size; the dict costs ~21MB RSS
- Built **CPU-only in 101 seconds** (1.94ms/word batched across cores) via `server/precompute.py`. CTranslate2 saturates the cores for the batch job, so even the precompute originally planned for the GPU turned out not to need one
- Ships uncommitted (gitignored, generated by `precompute.py`); the server loads it at startup
- **Browser offline dictionary:** the top **10,000** head words are bundled into the frontend (`demo/public/client_dict_hi.json`) as the offline fallback. Measured on held-out natural test traffic (not the corpus it was selected from), it covers **80.72%** of token volume: below the full server dict's 89.47%, but enough to keep the demo usable when the backend is unreachable. Reproduce with `scripts/zipf_coverage.py` (row `client_dict_offline`)

### 3.3 API Design

- `GET /transliterate?word=...&lang=...&topk=...`
- Response includes a `source` field (dict/cache/model) and server-measured `latency_ms`
- **Request coalescing (single-flight)** is implemented (`server/singleflight.py`, unit-tested): concurrent misses for the same word share one inference, so a suddenly-trending legal name cannot stampede the model. Coalesced followers are currently attributed to `cache` in `/metrics`; a distinct `coalesced` label would sharpen observability
- **Caching is application-owned, not HTTP/CDN-based:** responses are served `Cache-Control: no-store`, and the caching benefit comes from the server dictionary + LRU plus the browser session LRU + offline dictionary. This avoids external caches holding results that a model or dictionary redeploy would invalidate. CDN edge caching remains available as a scaling lever (section 7.1) but is a deliberate non-default

### 3.4 Reproducibility

```bash
git clone https://github.com/akhilagrawal14/indic_transliteration_system
cd indic_transliteration_system
cp .env.example .env
./run.sh docker
curl "http://localhost:8000/transliterate?word=mera&lang=hi&topk=5"
```

Every table in this report has a named script under `scripts/`, `eval/`, or `cost/`; exact commands are in Appendix B.

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

Hardware: a 4-vCPU CPU instance for the CPU rows; the GPU row was measured separately on a machine with 1x NVIDIA L4 (the rejected path). Dictionary row is a 41,345-entry in-memory map built from the Dakshina train lexicon.

Four results drive every decision in this report:

1. **Quantization is not an optimization, it is the requirement.** The stock fairseq FP32 engine has a p95 of 107.8ms for a single word: it exceeds the entire 100ms end-to-end budget before a single network hop is added. CTranslate2 INT8 cuts p50 by 9.9x (73.34 to 7.39ms) and p95 by 9.3x.
2. **The GPU is slower than the CPU for this workload** (analysis in section 2.3). The GPU path is not carried into the load tests: the microbenchmark already shows it is slower here, so there is nothing to gain by driving it under concurrency.
3. **Threading buys almost nothing.** Going from 1 to 24 CPU threads improves p50 by only 8% (7.39 to 6.82ms). The model is too small to parallelize within a request. The correct configuration is 1 intra-op thread per worker with many workers, which maximizes throughput per box rather than shaving single-request latency. This is what makes the CPU sizing so favorable.
4. **The dictionary path is effectively free.** At ~0.4 microseconds per lookup it is ~18,000x faster than the CT2 model path and its cost rounds to zero. This is why serving 89.5% of traffic from the dictionary collapses the compute requirement.

### 4.2 Decode-Time Frontiers (Why INT8, Why Beam 5)

**Precision frontier.** Every CTranslate2 CPU compute type, swept through both the latency microbench and the full quality eval (4,442 Dakshina test inputs, beam 5, 1 CPU thread). Reproduce with `python scripts/frontier.py --sweep precision`.

| compute_type | p50 (ms) | p95 (ms) | Top-1 | Top-5 | CER |
|---|---|---|---|---|---|
| **int8 (chosen)** | **7.5** | 11.8 | 61.08% | **87.30%** | 0.1183 |
| int8_float32 | 7.5 | 11.8 | 61.08% | 87.30% | 0.1183 |
| int16 | 11.3 | 17.4 | 61.23% | 87.30% | 0.1174 |
| float32 | 19.4 | 27.3 | 61.17% | 87.30% | 0.1176 |

INT8 is Pareto-optimal. Top-5 accuracy, the product metric (the typist picks from the ranked list), is identical at 87.30% at every precision, so there is zero product-relevant quality to buy with more latency. Higher precision costs 1.5x to 2.6x the latency for at most 0.15pp of top-1. And the INT8 step itself is nearly lossless: CT2 float32 scores 61.17% top-1 versus int8's 61.08%, so the remaining ~0.18pp gap to the fairseq FP32 reference (61.35%, section 5.1) comes from the CT2 conversion and its C++ beam search, not from quantization. (bfloat16/float16 are absent because CTranslate2 supports them only on GPU, and the CPU instance used predates AVX-512-BF16/AMX; noted as future work in section 7.2.)

**Beam width frontier.** Fixed INT8, sweeping the other decode knob. Reproduce with `python scripts/frontier.py --sweep beam`.

| beam | p50 (ms) | Top-1 | Top-k accuracy | candidates (k) |
|---|---|---|---|---|
| 1 (greedy) | 4.7 | 60.87% | 60.87% | 1 |
| 3 | 6.6 | 60.60% | 82.15% | 3 |
| **5 (chosen)** | 7.5 | 61.08% | **87.30%** | 5 |
| 8 | 8.8 | 61.23% | 87.53% | 5 |

Beam width caps the number of candidates, so beam < 5 cannot serve a top-5 dropdown at all (greedy returns a single spelling). Beam 5 is the minimum that fills the ranked list; beam 8 buys only +0.23pp top-5 for +17% latency. Beam 5 is the right operating point.

### 4.3 Load Test: Target Scale

Load generated with Locust: 500 simulated typists, Zipfian word distribution over Dakshina's natural romanized test corpus (held out from the dictionary), ~0.7 lookups/sec/user. The server was pinned to 4 cores (`taskset -c 0-3`, 4 uvicorn workers, 1 CT2 thread each) to approximate an n2-standard-4; Locust ran on the remaining cores. This is a CPU-capacity experiment, not a full-instance benchmark (no TLS, load balancer, network RTT, or GCP scheduler quota). Percentiles are end to end (Locust client side); server-side model latency is from `/metrics`.

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

1. **The chosen config clears the bar with enormous margin.** INT8 + dictionary holds p95 = 12ms end to end at 393 RPS on 4 cores, against a target of p95 < 100ms (and an internal server-side bar of 50ms). That is one n2-standard-4-class box doing ~1.3x the realistic peak with room to spare and zero errors.
2. **FP32 collapses under load even with the same 90% cache.** p95 balloons to 6.2 seconds. The cause is structural: a fairseq FP32 inference takes ~76ms and holds the worker (Python-side beam search), so the 9.5% of requests that miss the cache block the workers and the fast dictionary hits queue behind them. Aggregate throughput falls to 215 RPS because closed-loop users stall waiting on multi-second responses. This is the load-test proof that quantization is a requirement: single-word FP32 latency already exceeded budget (section 4.1), and under concurrency it becomes catastrophic.
3. **INT8 is the load-bearing component; the dictionary is the margin.** With the cache off entirely, INT8 still sustains 384 RPS at p95 = 54ms (server-side model p95 = 34.6ms). So the quantized CPU model alone nearly meets the target; the dictionary then drops model-path load ~12x, cuts p50 from 18ms to 2ms, and, more importantly, frees the headroom that lets one small box absorb 10x growth (next section).

### 4.4 Burst, Sustained Load, and the Ceiling

Further INT8 runs on the same 4-core config characterize the operating envelope. The two Sustained rows are 5-minute steady-state runs (ramp to a fixed user count, then hold), added to confirm the burst numbers persist over time:

| Run | Offered load | RPS | p50 | p95 | p99 | p99.9 | max | errors |
|---|---|---|---|---|---|---|---|---|
| Target (baseline) | 500 users @ ~0.7 rps | 393 | 2 | 12 | 20 | 49 | 82 | 0% |
| + jemalloc allocator | 500 users @ ~0.7 rps | 393 | 2 | 13 | 21 | 50 | 86 | 0% |
| Burst (step) | 1000 users all at once | 788 | 6 | 48 | 190 | 780 | 870 | 0% |
| **Sustained (5 min)** | **1000 users @ ~0.7 rps** | **782** | **6** | **35** | **66** | **130** | **260** | **0%** |
| **Sustained (5 min)** | **1300 users @ ~0.7 rps** | **906** | **110** | **300** | **400** | **480** | **580** | **0%** |
| Pessimistic (per-keystroke) | 500 users @ ~3.8 rps | 949 | 200 | 270 | 300 | 330 | 460 | 0% |

Server-side resource use under load (4 uvicorn workers, sampled mid-run via `/proc`):

| Sustained run | Server CPU (of 400% = 4 cores) | Server RSS (all workers) | Hit ratio dict/cache/model | Server-side model p99 |
|---|---|---|---|---|
| 1000 users, 782 RPS | **~130% (~1.3 cores)** | ~1.8 GB | 89.4% / 3.0% / 7.6% | 13.2 ms |
| 1300 users, 906 RPS | **~108% (~1.1 cores)** | ~1.8 GB | 89.5% / 5.3% / 5.2% | 11.0 ms |

- **Burst tolerance.** Spawning 1000 users instantly (2x target, ~788 RPS) is absorbed with zero errors; p95 stays at 48ms, and only p99/p99.9 briefly spike (190/780ms) before settling. The system degrades gracefully under a courtroom-session-start thundering herd.
- **Sustained load holds.** Held at 1000 concurrent typists (~782 RPS) for 5 minutes: p95 35ms, p99 66ms, 0 errors, comfortably inside budget. The four workers consumed only ~1.3 of 4 cores and RSS held at ~1.8GB, so at nearly 2x the realistic peak the box is about one-third utilized.
- **The measured ceiling is the load generator, not the server.** At 1300 users (~906 RPS) end-to-end p50 rose to 110ms, but server-side latency stayed at p99 <= 14ms and CPU at ~1.1 cores, while single-process Locust logged `CPU usage above 90%`. The added latency is queueing inside the load generator. The ~950 RPS figure from the pessimistic run is therefore a single-generator limit, not the server's ceiling: the 4-core server was never saturated in any cached run (it peaks at ~1.3 cores). Establishing the true server ceiling needs a distributed load generator; until then, ~800 RPS/box is a conservative, verified-sustained floor.
- **Allocator (jemalloc) buys nothing here.** `LD_PRELOAD=libjemalloc` reproduces the baseline within noise (p95 13 vs 12ms). An honest negative result: at INT8's low allocation churn the allocator is not the bottleneck, so it is not adopted. Checked rather than assumed.

**Cache-off: the model path's real ceiling, and proof the dictionary is load-bearing.** Disabling both the dictionary and the LRU so ~99% of requests hit CT2 (single-flight still coalesces concurrent duplicates, hence a few percent show as `cache`) moves the bottleneck from the load generator to the server itself. Sustained 5-minute cache-off runs on the same 4-core box:

| Cache-off run | RPS | p50 | p95 | p99 | max | Server CPU (of 400%) | Server-side model p99 | errors |
|---|---|---|---|---|---|---|---|---|
| 500 users | 390 | 20 | 55 | 87 | 350 | **~323% (~3.2 cores)** | 69 ms | 0% |
| 800 users | 490 | 270 | 1000 | 1200 | 1600 | **~398% (4 cores, pegged)** | 425 ms | 0% |

At 500 users the pure-model path is still in budget (390 RPS, p95 55ms) but already burns ~3.2 of 4 cores. At 800 users the cores peg and end-to-end p95 blows past budget to ~1000ms: genuine server saturation, so the pure-model sustainable ceiling is ~400 RPS/box. This is the architecture's central result in one contrast: **the same 4-core box that saturates at ~400 model-RPS serves 782+ RPS at only ~1.3 cores once the dictionary absorbs ~90% of traffic.** The dictionary is not an optimization layered on a fast model; it is what turns "one box per ~400 RPS" into "one box per several thousand RPS," which is what flattens the cost curve in section 6.

**Methodology notes.** Closed-loop users (each waits for a response before sending the next), so a saturated server yields fewer requests rather than errors; that is why the pessimistic and FP32 runs show high latency at 0% failures. No client-side timeout was set; in production those slow responses would be user-abandoned. Locust is single-process (one core for its event loop) and became the bottleneck above ~800 RPS.

---

## 5. Suggestion Quality vs the Baseline

### 5.1 Dakshina Hindi Test Set

Evaluated on all 4,442 unique romanized inputs in the Dakshina Hindi test lexicon, beam width 5, top-5. The lexicon is many-to-one (one romanization can have several acceptable native spellings), so predictions are scored against the set of acceptable forms; scoring against a single reference would understate accuracy. Reproduce with `eval/eval.py` and `eval/compare_results.py`.

| Engine | Top-1 Accuracy | Top-5 Accuracy | CER (top-1) |
|---|---|---|---|
| FP32 reproduction (stock fairseq) | 61.35% | 87.35% | 0.1170 |
| CTranslate2 INT8 (deployed) | 61.08% | 87.30% | 0.1183 |
| **Delta (INT8 vs FP32)** | **-0.27 pp** | **-0.05 pp** | **+0.0013** |

The quantization cost is negligible: top-1 within 0.27 percentage points, top-5 within 0.05. Top-5 is the closest available product proxy, since the typist picks from the ranked dropdown, and it is effectively unchanged. It is only a proxy, though: rank matters (a rank-5 pick costs more interaction than rank-1), so a fuller product evaluation would add MRR, mean accepted rank, and keystrokes-saved from real accept logs.

**Comparison to published numbers.** The IndicXlit paper reports higher Dakshina figures, but on a different protocol (its own held-out split with rescoring enabled). This evaluation runs with `rescore=False` for a clean FP32-vs-INT8 comparison on identical inputs, so the absolute figures are not directly comparable to the paper. The delta between engines, which is what the deployment decision hinges on, is the number that matters here.

**Scope caveat.** This measures benchmark-quality preservation: that the CT2 INT8 conversion did not regress the baseline model on Dakshina. It is not courtroom-domain validation. A production sign-off needs a domain eval set (judge/lawyer/party/village names, legal terminology, English/Hindi code-mixing, abbreviations and case identifiers, noisy typist romanization, digits/punctuation, Unicode/nukta normalization). That set does not exist yet and is the top item in section 7.2.

### 5.2 Qualitative Notes

The two engines produce the **identical top-1 candidate for 98.81%** of inputs. Top-1 changed on only 53 of 4,442 inputs, and the change was a net wash: 3 became correct, 3 became incorrect, the rest were reorderings among already-acceptable spellings.

Every top-1 disagreement inspected is a rank-1/rank-2 swap between two plausible transliterations, not a corruption:

| Input | FP32 top-2 | INT8 top-2 | Assessment |
|---|---|---|---|
| `aram` | आराम, अरम | अरम, आराम | both valid, reordered |
| `arya` | आर्य, आर्या | आर्या, आर्य | both valid, reordered |
| `chadron` | चद्रों, चादरों | चादरों, चद्रों | both valid, reordered |
| `dakhile` | दखिले, दाखिले | दाखिले, दखिले | both valid, reordered |

This is the expected signature of INT8 plus a C++ beam search: near-identical rankings with occasional adjacent swaps where two candidates score within rounding of each other. There is no systematic regression. Full per-word predictions are retained locally (`eval/results/*.json`, gitignored); the committed `*.summary.json` files carry the metrics and a sample of misses.

### 5.3 Is More Quantization Tuning Worth It?

No, and the frontier in section 4.2 shows why with numbers. The measured cost of the dynamic INT8 PTQ that CT2 applies is 0.27pp top-1 and 0.05pp top-5, of which only ~0.09pp is the INT8 step itself (CT2 float32 is already at 61.17% vs int8's 61.08%); the rest is the CT2 conversion. Top-5, the metric users experience, does not move at all. The remaining levers and why each is declined:

- **Higher-precision CPU compute types:** measured in section 4.2, they recover at most 0.15pp top-1 and 0.00pp top-5 at 1.5x to 2.6x the latency. Rejected
- **Calibration-based static INT8 PTQ:** the available headroom is ~0.09pp, and the ONNX comparison (section 2.4) settles it empirically: a different runtime's INT8 matches FP32 to within sampling noise, so quantization is lossless across engines and calibration has nothing to recover. Not pursued
- **QAT / distillation / retraining:** out of scope by assignment (the model is given), and pointless given the headroom
- **bf16/fp16:** not available in CTranslate2's CPU backend and the benchmark CPU predates AVX-512-BF16/AMX; noted as future work (section 7.2), no numbers claimed

INT8 sits at the knee of the curve: fastest option, quality within measurement noise of full precision on the product metric.

---

## 6. Cost Model

### 6.1 Assumptions

| Parameter | Value | Source |
|---|---|---|
| DAU | 5,000 | Target scale |
| Court hours/day | 3 | Typical Indian courtroom schedule |
| Lookups/user/hour | 1,500 | ~0.7 RPS x 3600s, with active duty cycle |
| Cache hit ratio | **92%** | Measured in the target load test (section 4.3) |
| Peak RPS (1x) | ~210 | 10% of DAU concurrent x 0.7 rps x 0.6 duty |
| Model-path RPS/core | 100 | Derated from the microbench ceiling (~135 QPS/core) |
| Per-box in-budget throughput | ~800 RPS | Verified sustained, server only ~1/3 utilized (section 4.4), so conservative |
| CPU instance | n2-standard-4 @ **$144.79/mo** | GCP on-demand, 4 vCPU / 16 GB / 30 GB |
| GPU instance | g2-standard-4 @ **$518.99/mo** | GCP on-demand, 1x L4 |
| HA floor | 2 instances | Behind an HTTPS load balancer |

Reproduce with `python cost/costmodel.py --hit-ratio 0.92`. Output: `cost/results/cost_comparison.json`.

### 6.2 Cost at Scale

| Scale | DAU | Peak RPS | Model RPS | CPU + Dict ($/mo) | GPU Always-On ($/mo) |
|---|---|---|---|---|---|
| 1x (current) | 5,000 | 210 | 17 | **289.58** (2x n2) | 1,037.98 (2x g2) |
| 3x | 15,000 | 630 | 50 | **289.58** (2x n2) | 1,037.98 (2x g2) |
| 10x | 50,000 | 2,100 | 168 | **434.37** (3x n2) | 1,556.97 (3x g2) |

The CPU line is nearly flat: at 1x and 3x the 2-instance HA floor dominates, since peak RPS is well under one box's verified ~800 RPS in-budget floor. Only at 10x does aggregate throughput require a third box (2,100 / 800, rounded up to 3). Note the binding constraint at scale is aggregate HTTP throughput per box (measured in section 4.4), not the model path, which is only 168 RPS at 10x. The GPU line is **3.6x to 5.4x more expensive at every point**. This flat-vs-linear gap is the core economic argument, and the recommended architecture sits 42% under the ~$500/month ceiling at target scale even with full HA.

### 6.3 Cost per Unit

| Metric | CPU + Dict (1x / 10x) | GPU (1x / 10x) |
|---|---|---|
| $/1k users/month | $57.92 / $8.69 | $207.60 / $31.14 |
| $/1M lookups | $0.43 / $0.06 | $1.54 / $0.23 |

Both curves fall with scale as fixed cost amortizes, but CPU is ~3.6x cheaper per unit throughout. At 10x, the CPU path serves a courtroom user for a fraction of a cent per day.

### 6.4 One-Time Experiment Cost

The GPU path was benchmarked on a machine with one L4 and rejected; all serving and CPU benchmarks then ran on a CPU instance. The GPU test is a one-time evaluation cost, not recurring. The recurring serving cost is the $289.58/month above.

---

## 7. Key Takeaways, Scaling, and Hardening

### 7.1 What Breaks at 10x, and the Migration Path

At 10x (50k DAU, ~2,100 peak RPS), the model path is a non-issue (~168 RPS); aggregate HTTP throughput becomes the sizing constraint, needing 3 instances (~$434/month), still 3.6x cheaper than the GPU path. What actually breaks first, in order:

1. **Single-region RTT.** For a pan-India deployment, the network dominates the 100ms budget (section 1.2). Fix: CDN edge caching (GET responses are deterministic and cacheable; currently a deliberate non-default per section 3.3, but available as a lever with cache-versioned URLs) and/or multi-region instances
2. **Dictionary staleness.** New names, legal terms, and neologisms miss the precomputed set. Fix: a nightly job that promotes frequent model-path misses into the dictionary (a data flywheel), plus periodic re-precompute as vocabulary grows. The 101-second CPU-only precompute makes this cheap
3. **Observability.** Aggregate p95 hides per-courtroom tail latency. Fix: Prometheus histograms with per-source labels replacing the debug `/metrics`, and per-region/per-courtroom dashboards. Cache hit ratio is the single most important production metric
4. **The worker-blocking failure mode.** The FP32 collapse (section 4.3) is a general warning: any slow synchronous call on the model path head-of-line-blocks a worker. Keep the model path fast, and use a separate worker pool or async offload if a heavier model is ever introduced

The migration path is additive, not a rewrite: same API, add CDN, add regions, grow the dictionary.

### 7.2 What I Would Do Next

Validation and hardening, in priority order:

- **Courtroom-domain evaluation set** (highest priority). Names, legal terminology, code-mixing, abbreviations, noisy romanization, normalization edge cases; report MRR, mean accepted rank, and keystrokes-saved from real accept logs, not just Dakshina top-5. Until then the quality claim is benchmark preservation, not product fitness
- **True server ceiling + real-instance testing.** The sustained runs confirmed ~800 RPS/box in-budget at ~1/3 utilization, but also showed the single-process load generator caps measurement past that. Remaining work: a distributed load generator to find where the server itself saturates, plus a real n2-standard-4 behind TLS and a load balancer
- **Statistical rigor.** Paired bootstrap or McNemar on the paired predictions instead of the informal "within noise" (the 98.81% identical-top-1 and 3-worse/3-better split already point the same way)
- **Offline hardening.** On-device WASM inference and a service-worker cache for true offline courtroom operation (the 13MB INT8 model makes this plausible); personalized accept-history ranking; streaming prefix suggestions
- **Runtime frontier on newer CPUs.** bf16/AMX-INT8 on Sapphire Rapids (C3) could cut model-path latency further; documented, not measured

### 7.3 What Was Traded Away

- No GPU anywhere (justified by measurement: slower and more expensive for this workload)
- Fixed decode parameters (beam 5, top-5), chosen from the measured frontier; beam width must be >= topk or candidates are silently truncated
- Single-language deep dive (Hindi); the architecture is language-agnostic but each language needs its own dictionary precompute
- No auth, rate limiting, or multi-tenancy in the prototype
- Minimal frontend (functional demo, not a product)

---

## Appendix

### A. Hardware and Environment

| Role | Spec | Notes |
|---|---|---|
| CPU serving + benchmarks | 4-vCPU Linux instance (n2-standard-4 class), 4 uvicorn workers, 1 CT2 thread each | All serving, precompute, eval, and load tests |
| GPU comparison (rejected path) | 1x NVIDIA L4 | Used only to benchmark and reject the GPU option |
| OS / toolchain | Debian 11 (bullseye), Python 3.10.20 (conda env `xlit`) | fairseq 0.12.2, CTranslate2 4.8.1, torch 2.5.1 |
| Load generator | Separate host / dedicated cores (`taskset`) | Never competes with the server's 4 cores |

Microbenchmarks ran with the server not under concurrent load. Load tests pinned the server to 4 vCPUs with the generator isolated on the remaining cores. No GPU is used in serving, precomputation, or load testing.

### B. How to Reproduce Every Result

```bash
# Section 1.3: Zipfian coverage and held-out dictionary hit rate
python scripts/zipf_coverage.py --output eval/results/zipf_coverage.json

# Section 4.1: single-word latency percentiles across all engines
python scripts/microbench.py --output eval/results/microbench.json
python scripts/microbench.py --print-table eval/results/microbench.json

# Section 4.2: precision and beam frontiers
python scripts/frontier.py --sweep precision
python scripts/frontier.py --sweep beam

# Section 2.3: FLOP budget per request (from the checkpoint config)
python scripts/flops.py

# Section 2.4: ONNX export and engine comparison
python scripts/export_onnx.py
python scripts/onnx_compare.py

# Section 5: quality parity, FP32 baseline vs INT8
python eval/eval.py --engine fairseq --lang hi --topk 5 --output eval/results/baseline_fp32.json
python eval/eval.py --engine ct2 --lang hi --topk 5 --output eval/results/ct2_int8.json

# Sections 4.3-4.4: load tests (server pinned to cores 0-3, Locust on 4-23;
# a separate load-generator VM is equivalent)
XLIT_ENGINE=ct2 taskset -c 0-3 uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 4 &
taskset -c 4-23 locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --users 500 \
  --spawn-rate 50 --run-time 300s --headless --csv loadtest/results/scenario2_int8

# Section 6: cost model
python cost/costmodel.py --hit-ratio 0.92 --output cost/results/cost_comparison.json
```

Raw results are committed under `eval/results/`, `loadtest/results/`, and `cost/results/`.