# Indic Transliteration Runtime

A benchmarked prototype for Indic transliteration serving in Indian courtrooms. Converts Latin input (e.g. `mera`) into ranked Devanagari candidates (`मेरा`, `मीरा`, `मैरा`, ...) and meets p95 < 100ms at 500 concurrent simulated users in load tests. See the [report](report/report.md) for scope, evidence, and the remaining work to production-harden it.

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

**Why not a GPU:** at ~300 RPS of single-word character-level requests, an always-on L4 sits at low single-digit utilization while costing ~$519/month per instance, and it is actually slower than one CPU core at batch size 1 (measured). The CPU + dictionary path is ~$145/month per instance ($290 for a 2-instance HA pair) and, because the dictionary absorbs growth, stays flat to ~3x and needs only a third instance at ~10x.

---

## Quick Start

### Prerequisites
- Python 3.10 (via conda) for the backend
- Node.js 20 (for the frontend demo)
- Docker + Docker Compose (for Option A)
- No GPU needed. Serving, precompute, eval, and load tests are all CPU-only.

### One-time setup (both options)
```bash
git clone <repo-url> && cd indic_transliteration_system
cp .env.example .env

# Build artifacts on first run: dataset, model, CT2 conversion, dictionary,
# and the demo's offline dictionary (~a few minutes, CPU-only).
conda activate xlit
bash scripts/bootstrap.sh
```
See [docs/setup.md](docs/setup.md) for the full environment setup (conda env, deps).

### Option A — Docker (one command)
Builds and runs backend + frontend together:
```bash
docker compose up --build
# Backend API:  http://localhost:8000
# Demo (UI):    http://localhost:3000
```

### Option B — Local dev (two terminals)
```bash
# Terminal 1: backend
conda activate xlit
XLIT_ENGINE=ct2 uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: frontend
cd demo
npm install       # first time only
npm run dev       # http://localhost:3000
```
`./run.sh docker` and `./run.sh dev` are shortcuts for the two options above.

### Verify
```bash
# Backend
curl "http://localhost:8000/transliterate?word=mera&topk=5"
# {"input":"mera","candidates":["मेरा","मीरा","मैरा",...],"source":"dict","latency_ms":0.3}
curl "http://localhost:8000/healthz"     # {"status":"ok","engine":"ct2","dict_size":52045}
```
Then open **http://localhost:3000**, type romanized Hindi (e.g. `mera nyayalaya`), and pick from the ranked suggestions. If the backend is unreachable, head words still resolve from the demo's bundled offline dictionary.

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

Measured on a 4-vCPU CPU instance (n2-standard-4 class). The GPU path was benchmarked separately on a GPU machine (1x NVIDIA L4) and rejected; serving is CPU-only. Full detail and methodology in the [report](report/report.md).

### Latency

| Path | p50 | p95 | p99 |
|---|---|---|---|
| Dictionary hit (single word) | ~0.0004 ms | ~0.001 ms | ~0.001 ms |
| Model, CT2 INT8 CPU (single word) | 7.5 ms | 11.8 ms | 13.7 ms |
| End-to-end, 500 users mixed (load test) | 2 ms | 12 ms | 20 ms |
| FP32 baseline, 500 users (why quantize) | 370 ms | 6,200 ms | 9,900 ms |

### Quality (Dakshina Hindi test set, 4,442 inputs)

| Engine | Top-1 | Top-5 | CER |
|---|---|---|---|
| IndicXlit FP32 (baseline) | 61.35% | 87.35% | 0.117 |
| CTranslate2 INT8 (chosen) | 61.08% | 87.30% | 0.118 |
| Delta | -0.27 pp | -0.05 pp | +0.001 |

Note: benchmark-quality preservation on Dakshina, not courtroom-domain validated (see report §5).

### Cost at Scale (CPU + dictionary vs always-on GPU)

| Architecture | 5k DAU | 15k DAU | 50k DAU |
|---|---|---|---|
| CPU + dictionary (chosen) | $290/mo | $290/mo | $434/mo |
| Always-on GPU (rejected) | $1,038/mo | $1,038/mo | $1,557/mo |

Figures are the recommended 2-instance HA configuration (single-instance compute is roughly half). Excludes load balancer, CDN, egress, and monitoring; see report §6.

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