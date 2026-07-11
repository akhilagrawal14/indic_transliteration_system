# Environment Setup Guide

Step-by-step setup for the Indic Transliteration Runtime project on a Debian-based GCP g2-standard-24 instance with 2x L4 GPUs.

---

## Prerequisites Checklist

Before starting, confirm you have:

- [ ] GCP g2-standard-24 instance running (Debian 11/12)
- [ ] SSH access to the instance
- [ ] At least 100GB disk space (models + datasets + conda)
- [ ] NVIDIA drivers installed (verify with `nvidia-smi`)
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

# 1.5 Verify NVIDIA driver and CUDA
nvidia-smi
# Expected: 2x NVIDIA L4, Driver 535+, CUDA 12.x
# If missing, install: sudo apt install -y nvidia-driver-535
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
pip-compile requirements.in -o requirements.txt --resolver=backtracking

# 3.4 Install all dependencies
pip install -r requirements.txt

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
wc -l eval/data/dakshina/hi/lexicons/hi.translit.sampled.*.tsv
# Should show thousands of lines per split

# 5.3 Preview the format
head -5 eval/data/dakshina/hi/lexicons/hi.translit.sampled.test.tsv
# Format: native_script \t romanized \t count
# Example: मेरा    mera    42

# 5.4 Download Aksharantar word list (for dictionary precomputation)
# This is AI4Bharat's larger transliteration dataset
python scripts/download_aksharantar.py

# 5.5 Verify
ls -lah eval/data/aksharantar/
wc -l eval/data/aksharantar/hi/*.txt
```

---

## Phase 6: Model Conversion (CTranslate2 INT8 + ONNX INT8)

This is where your L4 GPUs earn their keep. Conversion runs once; serving uses the output.

```bash
# 6.1 Convert fairseq checkpoint to CTranslate2 INT8
# If this fails, skip to 6.2 (ONNX fallback)
python scripts/convert_ct2.py \
    --model-dir models/indicxlit/fairseq_original \
    --output-dir models/indicxlit/ct2_int8 \
    --quantization int8 \
    --lang hi

# 6.2 Verify CT2 conversion
python -c "
import ctranslate2
translator = ctranslate2.Translator('models/indicxlit/ct2_int8', device='cpu')
print('CT2 model loaded successfully')
print(f'Device: {translator.device}')
"

# 6.3 (FALLBACK) Convert to ONNX INT8 if CT2 fails
python scripts/convert_onnx.py \
    --model-dir models/indicxlit/fairseq_original \
    --output-dir models/indicxlit/onnx_int8 \
    --quantize int8 \
    --lang hi

# 6.4 Run quality check: compare FP32 vs quantized on a sample
python eval/eval.py \
    --engine fairseq \
    --lang hi \
    --data eval/data/dakshina/hi/lexicons/hi.translit.sampled.test.tsv \
    --topk 5 \
    --output eval/results/baseline_fp32.json

python eval/eval.py \
    --engine ct2 \
    --lang hi \
    --data eval/data/dakshina/hi/lexicons/hi.translit.sampled.test.tsv \
    --topk 5 \
    --output eval/results/ct2_int8.json

# 6.5 Compare: should show near-zero quality delta
python eval/compare_results.py \
    --baseline eval/results/baseline_fp32.json \
    --candidate eval/results/ct2_int8.json
```

---

## Phase 7: Dictionary Precomputation

Use the GPU to bulk-transliterate the full word list. This is the key to the architecture: precompute once, serve from memory forever.

```bash
# 7.1 Precompute dictionary (runs on GPU, takes ~10-30 min depending on vocab size)
CUDA_VISIBLE_DEVICES=0 python server/precompute.py \
    --wordlist eval/data/dakshina/hi/lexicons/hi.translit.sampled.train.tsv \
    --aksharantar eval/data/aksharantar/hi/ \
    --lang hi \
    --beam-width 4 \
    --topk 5 \
    --output server/data/dictionary_hi.json \
    --format json

# 7.2 Check dictionary size and sample entries
python -c "
import json
with open('server/data/dictionary_hi.json') as f:
    d = json.load(f)
print(f'Dictionary entries: {len(d)}')
print(f'File size: {__import__(\"os\").path.getsize(\"server/data/dictionary_hi.json\") / 1024 / 1024:.1f} MB')
# Show a few entries
for k in list(d.keys())[:5]:
    print(f'  {k} -> {d[k]}')
"

# 7.3 Build compressed trie (marisa-trie) for memory-efficient serving
python server/build_trie.py \
    --input server/data/dictionary_hi.json \
    --output server/data/dictionary_hi.marisa
```

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

## Phase 12: Load Testing

```bash
# 12.1 Start the server (if not running)
# In terminal 1:
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

# 12.2 Run Locust (in a separate terminal, ideally a separate machine)
cd ~/indic-xlit-runtime/loadtest
locust -f locustfile.py --host http://localhost:8000

# 12.3 Open Locust UI at http://localhost:8089
# Configure: 500 users, spawn rate 50/s, run for 10 minutes

# 12.4 Or run headless for the report numbers:
locust -f locustfile.py \
    --host http://localhost:8000 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 600s \
    --headless \
    --csv loadtest/results/run_500users

# 12.5 GPU comparison run (start server on GPU for this run only)
DEVICE=cuda uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1
locust -f locustfile.py \
    --host http://localhost:8001 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 300s \
    --headless \
    --csv loadtest/results/run_500users_gpu
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
python eval/eval.py --engine ct2 --lang hi --data eval/data/dakshina/hi/lexicons/hi.translit.sampled.test.tsv

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