# Environment Setup Guide

Step-by-step setup on a Debian-based Linux CPU instance. Serving, precompute, eval, and load tests are all CPU-only. A GPU machine (e.g. 1x L4) is optional and only needed to reproduce the rejected-path GPU comparison; skip every GPU/CUDA step below if you don't need that.

---

## Prerequisites Checklist

Before starting, confirm you have:

- [ ] A Linux CPU instance (Debian 11/12), 4+ vCPUs
- [ ] SSH access to the instance
- [ ] At least 100GB disk space (models + datasets + conda)
- [ ] (Optional) NVIDIA GPU + drivers, only to reproduce the GPU comparison
- [ ] Git installed (`sudo apt install git`)
- [ ] OpenAI API key (optional, for LLM-as-judge eval only)

---

## Phase 1: System-Level Dependencies

Run these on a fresh Debian instance. Skip any step where the tool is already present.

```bash
# 1.1 Update system packages
sudo apt update && sudo apt upgrade -y

# 1.2 Install build essentials and dev headers
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    wget \
    curl \
    unzip \
    git \
    htop \
    tmux \
    jq

# 1.3 Install Docker and Docker Compose (for containerized deploy)
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and back in for group to take effect, or:
newgrp docker

# 1.4 Install Node.js 20 LTS (for Next.js demo)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version  # should show v20.x
npm --version

# 1.5 (OPTIONAL, GPU comparison only) Verify NVIDIA driver and CUDA
nvidia-smi
# CPU-only serving does not need this; skip unless reproducing the GPU benchmark.
```

---

## Phase 2: Conda Environment

We use Python 3.10 specifically because fairseq and the ai4bharat-transliteration library have known compatibility issues with 3.11+. Do NOT use 3.11 or 3.12.

```bash
# 2.1 Install Miniconda (if not present)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# 2.2 Create project environment
conda create -n xlit python=3.10 -y
conda activate xlit

# 2.3 Install CUDA toolkit inside conda (for PyTorch GPU support)
conda install -c conda-forge cudatoolkit=12.1 -y

# 2.4 Verify
python --version   # 3.10.x
which python        # should point to conda env
```

---

## Phase 3: Python Dependencies

We use pip-compile to lock versions. The requirements.in is the source of truth.

```bash
# 3.1 Navigate to project root
cd ~/indic-xlit-runtime

# 3.2 Install pip-tools first
pip install pip-tools

# 3.3 Compile locked requirements (run this once; re-run if you edit requirements.in)
python -m piptools compile requirements/requirements.in -o requirements/requirements.txt --resolver=backtracking

# 3.4 Install all dependencies
pip install -r requirements/requirements.txt
# Offline eval + precompute deps (fairseq/ai4bharat) are separate; see
# requirements/requirements-precompute.in for the install steps.

# 3.5 Verify critical imports
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import ctranslate2; print(f'CTranslate2 {ctranslate2.__version__}')"
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python -c "import locust; print(f'Locust {locust.__version__}')"
```

**If torch.cuda.is_available() returns False:**
```bash
# Reinstall PyTorch with CUDA 12.1 wheels explicitly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

**If fairseq install fails** (common issue):
```bash
# fairseq from source as fallback
pip install fairseq==0.12.2
# If that also fails:
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2
# Nuclear option: install from the ai4bharat fork which patches fairseq
pip install ai4bharat-transliteration
```

---

## Phase 4: Model Downloads

All models go into `models/` directory. We download everything upfront so no network calls are needed during serving or eval.

```bash
# 4.1 Create model directories
mkdir -p models/indicxlit/{fairseq_original,ct2_int8,onnx_int8}

# 4.2 Download IndicXlit model weights from AI4Bharat
# The ai4bharat-transliteration package auto-downloads on first use,
# but we want explicit control. Download from HuggingFace:
python scripts/download_models.py

# 4.3 Verify the download
ls -lah models/indicxlit/fairseq_original/
# Should contain: dict.*.txt, model checkpoints, sentencepiece models
# Total size: ~200-500MB depending on languages

# 4.4 Quick smoke test: run a single transliteration
python -c "
from ai4bharat.transliteration import XlitEngine
engine = XlitEngine('hi', beam_width=4, rescore=False)
result = engine.translit_word('mera', lang_code='hi', topk=5)
print(result)
# Expected: ['मेरा', 'मीरा', 'मैरा', ...]
"
```

---

## Phase 5: Dataset Downloads

All evaluation and dictionary-building data goes into `eval/data/`.

```bash
# 5.1 Download Dakshina dataset (Google's romanization lexicons)
python scripts/download_data.py

# 5.2 Verify downloads
wc -l eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.*.tsv
# Should show thousands of lines per split

# 5.3 Preview the format
head -5 eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv
# Format: native_script \t romanized \t count
# Example: मेरा    mera    42

# The dictionary is built from the Dakshina train lexicon + natural corpus
# vocabulary (see server/precompute.py). Aksharantar is not required.
```

**Shortcut:** `scripts/bootstrap.sh` runs Phases 5, 4, 6.1 and the dictionary
precompute in order for a fresh clone.

---

## Phase 6: Model Conversion (CTranslate2 INT8 + ONNX INT8)

This is where your L4 GPUs earn their keep. Conversion runs once; serving uses the output.

```bash
# 6.1 Convert fairseq checkpoint to CTranslate2 INT8
# (handles the --unsafe_deserialization + lang_list + source/target-lang flags)
python scripts/convert_ct2.py

# 6.2 Verify CT2 conversion
python -c "
import ctranslate2
translator = ctranslate2.Translator('models/indicxlit/ct2_int8', device='cpu')
print('CT2 model loaded successfully')
print(f'Device: {translator.device}')
"

# 6.3 (OPTIONAL) Export to ONNX for the runtime comparison experiment
# (CT2 succeeded, so ONNX is not a serving path; this reproduces report 4.1d)
python scripts/export_onnx.py

# 6.4 Run quality check: compare FP32 vs quantized on a sample
python eval/eval.py \
    --engine fairseq \
    --lang hi \
    --data eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv \
    --topk 5 \
    --output eval/results/baseline_fp32.json

python eval/eval.py \
    --engine ct2 \
    --lang hi \
    --data eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv \
    --topk 5 \
    --output eval/results/ct2_int8.json

# 6.5 Compare: should show near-zero quality delta
python eval/compare_results.py \
    --baseline eval/results/baseline_fp32.json \
    --candidate eval/results/ct2_int8.json
```

---

## Phase 7: Dictionary Precomputation (CPU only)

Bulk-transliterate the word list once, then serve from memory forever. No GPU is
needed: CTranslate2 saturates the CPU cores for the batch, so the full ~52k-word
dictionary builds in ~100 s.

```bash
# 7.1 Precompute dictionary (CPU; ~100 s for ~52k words)
python server/precompute.py \
    --model-dir models/indicxlit/ct2_int8 \
    --lang hi --beam 5 --topk 5 \
    --output server/data/dictionary_hi.json

# 7.2 Check dictionary size and sample entries
python -c "
import json, os
d = json.load(open('server/data/dictionary_hi.json'))
print(f'Dictionary entries: {len(d)}')
print(f'File size: {os.path.getsize(\"server/data/dictionary_hi.json\")/1024/1024:.1f} MB')
for k in list(d.keys())[:5]:
    print(f'  {k} -> {d[k]}')
"
```

Expected: ~52,045 entries, ~6.5 MB. The server loads this JSON into an in-memory
dict at startup (marisa-trie is available as a memory optimization but is
unnecessary at this size).

---

## Phase 8: Microbenchmarks (Single-Word Latency)

Run these BEFORE building the server. These numbers anchor the architecture decision.

```bash
# 8.1 Single-word latency across all engines and devices
python scripts/microbench.py \
    --engines fairseq_cpu fairseq_gpu ct2_cpu ct2_gpu dict_lookup \
    --lang hi \
    --words "mera" "nyayalaya" "adhikari" "bharat" "samvidhan" \
    --iterations 100 \
    --output eval/results/microbench.json

# 8.2 Print results table
python scripts/microbench.py --print-table eval/results/microbench.json

# Expected output (approximate):
# | Engine         | Device | p50 (ms) | p95 (ms) | p99 (ms) |
# |----------------|--------|----------|----------|----------|
# | fairseq        | CPU    | ~80      | ~130     | ~160     |
# | fairseq        | GPU-L4 | ~12      | ~18      | ~22      |
# | ct2 INT8       | CPU    | ~8       | ~18      | ~25      |
# | ct2 INT8       | GPU-L4 | ~3       | ~6       | ~8       |
# | dict lookup    | CPU    | ~0.01    | ~0.02    | ~0.03    |
```

---

## Phase 9: Backend Server Setup

```bash
# 9.1 Test the server locally (not in Docker yet)
cd ~/indic-xlit-runtime
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

# 9.2 In another terminal, smoke test
curl "http://localhost:8000/transliterate?word=mera&lang=hi&topk=5" | jq .
# Expected:
# {
#   "input": "mera",
#   "candidates": ["मेरा", "मीरा", "मैरा", ...],
#   "source": "dict",
#   "latency_ms": 0.3
# }

curl "http://localhost:8000/healthz"
# Expected: {"status": "ok", "engine": "ct2_int8", "dict_size": 85000}
```

---

## Phase 10: Frontend Demo Setup

```bash
# 10.1 Initialize the Next.js demo
cd ~/indic-xlit-runtime/demo
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --no-import-alias
# Accept defaults when prompted

# 10.2 Install additional dependencies
npm install

# 10.3 Create .env.local for the demo
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# 10.4 Run dev server
npm run dev
# Opens at http://localhost:3000
```

---

## Phase 11: Docker Setup

```bash
# 11.1 Build and run both services
cd ~/indic-xlit-runtime
docker compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000

# 11.2 Verify
curl http://localhost:8000/transliterate?word=mera&lang=hi&topk=5
```

---

## Phase 12: Load Testing (CPU only)

Pin the server to 4 cores so numbers map to an n2-standard-4, and run Locust on
the remaining cores so the load generator does not contaminate server latency
(isolation on a single box; a separate VM is fine too).

```bash
# 12.1 Start the server pinned to cores 0-3 (terminal 1)
XLIT_ENGINE=ct2 taskset -c 0-3 uvicorn server.app:app \
    --host 127.0.0.1 --port 8000 --workers 4

# 12.2 Scenario 2 (target scale) on the remaining cores (terminal 2)
taskset -c 4-23 locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
    --users 500 --spawn-rate 50 --run-time 300s --headless \
    --csv loadtest/results/scenario2_int8

# 12.3 FP32 comparison: restart the server with XLIT_ENGINE=fairseq, rerun 12.2
#      into loadtest/results/scenario2_fp32

# 12.4 Scenario 4 (cache-off ablation): restart with the dictionary disabled
XLIT_ENGINE=ct2 XLIT_DICT_PATH="" XLIT_LRU_CACHE_SIZE=0 taskset -c 0-3 \
    uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 4
taskset -c 4-23 locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
    --users 500 --spawn-rate 50 --run-time 180s --headless \
    --csv loadtest/results/scenario4_cacheoff_int8

# Server-side hit ratio and per-worker latency are at GET /metrics.
# No GPU run is needed: the microbenchmarks already show the L4 is slower here.
```

---

## Phase 13: Cost Model

```bash
# 13.1 Run cost model with your measured throughput numbers
python cost/costmodel.py \
    --cpu-latency-p95-ms 18 \
    --gpu-latency-p95-ms 6 \
    --cache-hit-ratio 0.92 \
    --dau 5000 \
    --output cost/results/cost_comparison.json

# 13.2 Generate the sensitivity table (1x, 3x, 10x users)
python cost/costmodel.py --sensitivity --output cost/results/sensitivity.json
```

---

## Phase 14: Git Initialization

```bash
cd ~/indic-xlit-runtime
git init
git add .
git commit -m "feat: initial project scaffold with setup docs"
```

---

## Quick Reference: Key Commands

```bash
# Activate environment
conda activate xlit

# Run server (dev)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4 --reload

# Run frontend (dev)
cd demo && npm run dev

# Run both via Docker
docker compose up --build

# Run eval
python eval/eval.py --engine ct2 --lang hi --data eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv

# Run load test
cd loadtest && locust -f locustfile.py --host http://localhost:8000

# Recompile requirements after editing requirements.in
pip-compile requirements.in -o requirements.txt --resolver=backtracking
```

---

## Troubleshooting

**fairseq won't install:**
```bash
# Try installing from the specific commit known to work with ai4bharat
pip install fairseq==0.12.2 --no-build-isolation
# If that fails, install ai4bharat-transliteration first (it pins its own fairseq)
pip install ai4bharat-transliteration
```

**CTranslate2 conversion fails with "unsupported model":**
The fairseq checkpoint format may not match what ct2-fairseq-converter expects. Check:
```bash
ct2-fairseq-converter --help
# Make sure you are pointing to the correct checkpoint file, not the directory
# The checkpoint is typically named model.pt or checkpoint_best.pt
```
If conversion truly fails, fall back to ONNX (Phase 6.3) or serve with plain PyTorch on CPU. The architecture (dictionary + model tail) works with any engine.

**CUDA out of memory during precomputation:**
```bash
# Use only one GPU and limit batch size
CUDA_VISIBLE_DEVICES=0 python server/precompute.py --batch-size 32
```

**Node.js version conflicts:**
```bash
# Use nvm to manage Node versions
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
```