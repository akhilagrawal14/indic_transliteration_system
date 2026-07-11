# Indic Transliteration Runtime

Production-grade Indic transliteration serving for Indian courtrooms. Converts Latin input (e.g. `mera`) into ranked Devanagari candidates (`मेरा`, `मीरा`, `मैरा`, ...) with p95 < 100ms at 500 concurrent users.

---

## Architecture

```
Typist client
  |
  +-- (1) Client-side LRU cache (session lookups, 0ms, works offline)
  |
  +-- (2) GET /transliterate?word=mera&lang=hi&topk=5
           |
           +-- (2a) Dictionary lookup: precomputed top-100k        [<1ms]
           |        romanized -> candidates map (in-memory)
           |
           +-- (2b) Hot LRU cache of recent model misses           [<1ms]
           |
           +-- (2c) CTranslate2 INT8 model on CPU                  [5-25ms]
                    (writes result back into 2b)
```

**Why this works:** courtroom Hindi vocabulary is Zipfian. The top ~100k romanized inputs cover 90%+ of real lookups. Transliteration is deterministic per input. So a precomputed dictionary serves the head in sub-millisecond time, and only the tail (rare names, unusual terms) hits the model.

**Why not a GPU:** at ~300 RPS of single-word character-level requests, a GPU sits at 3% utilization while costing $500-700/month. The CPU + dictionary path costs ~$150/month and handles 10x growth before needing a second instance.

---

## Quick Start

### Prerequisites
- Python 3.10 (via conda)
- Node.js 20 (for frontend demo)
- Docker + Docker Compose (for containerized deployment)
- NVIDIA GPU (for model conversion and benchmarking only, not serving)

### Setup
See [docs/setup.md](docs/setup.md) for the full step-by-step guide.

```bash
# Clone and enter
git clone <repo-url> && cd indic-xlit-runtime

# Copy environment template
cp .env.example .env

# Option 1: Docker (recommended)
./run.sh docker

# Option 2: Local dev
./run.sh dev
```

### Verify
```bash
curl "http://localhost:8000/transliterate?word=mera&lang=hi&topk=5"
# {"input":"mera","candidates":["मेरा","मीरा","मैरा",...],"source":"dict","latency_ms":0.3}

curl "http://localhost:8000/healthz"
# {"status":"ok","engine":"ct2_int8","dict_size":85000}
```

---

## Project Structure

```
indic-xlit-runtime/
├── server/           # FastAPI backend (dictionary + model serving)
│   ├── engine/       # Pluggable model engines (fairseq, ct2, onnx)
│   └── tests/
├── eval/             # Quality evaluation (Dakshina, top-1/top-5/CER)
├── loadtest/         # Locust load tests (Zipfian typist simulation)
├── cost/             # Cost model ($/month at 1x/3x/10x scale)
├── scripts/          # Download, convert, benchmark scripts
├── demo/             # Next.js courtroom editor demo
├── models/           # Model weights (gitignored, downloaded by scripts)
├── docs/             # Setup guide, architecture decisions
└── report/           # Final write-up
```

---

## Key Results

*(Fill in after benchmarking)*

### Latency (p50 / p95 / p99)

| Path | p50 | p95 | p99 |
|---|---|---|---|
| Dictionary hit | ms | ms | ms |
| LRU cache hit | ms | ms | ms |
| Model (CT2 INT8, CPU) | ms | ms | ms |
| End-to-end (mixed, 500 users) | ms | ms | ms |

### Quality (Dakshina Hindi test set)

| Engine | Top-1 Accuracy | Top-5 Accuracy | CER |
|---|---|---|---|
| IndicXlit FP32 (baseline) | % | % | |
| CTranslate2 INT8 | % | % | |
| Delta | % | % | |

### Cost at Scale

| Architecture | 5k DAU | 15k DAU | 50k DAU |
|---|---|---|---|
| CPU + dictionary (chosen) | $/mo | $/mo | $/mo |
| Always-on GPU (rejected) | $/mo | $/mo | $/mo |

---

## Deliverables

- [Report](report/report.md) -- workload characterization, architecture decision, benchmarks, cost model, reflections
- [Codebase](.) -- modular, documented, Dockerized
- [Benchmark results](eval/results/) -- latency percentiles, quality metrics
- [Cost model](cost/results/) -- $/month with sensitivity analysis
- [Demo](demo/) -- courtroom editor with live transliteration suggestions

---

## Tech Stack

- **Backend:** Python 3.10, FastAPI, CTranslate2, marisa-trie
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Load Testing:** Locust
- **Containerization:** Docker, Docker Compose
- **Model:** AI4Bharat IndicXlit (fairseq transformer, INT8 quantized)

---

## Author

Akhil Agrawal -- [akhilagrawal@gmail.com](mailto:akhilagrawal@gmail.com)