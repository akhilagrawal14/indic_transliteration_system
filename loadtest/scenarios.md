# Load Test Scenarios

All scenarios use a Zipfian word distribution over the Dakshina Hindi vocabulary to simulate realistic courtroom typing. Words are drawn from the test set with frequency weighting so common words (like "mera", "hai", "ka") appear far more often than rare ones.

---

## Scenario 1: Latency Floor (Single User)

**Purpose:** Establish the minimum achievable latency with zero contention.

| Parameter | Value |
|---|---|
| Users | 1 |
| Duration | 60s |
| Request pattern | Sequential (wait for response before next) |
| Cache state | Run twice: cold (empty LRU) and warm (pre-filled) |
| Metrics to capture | p50, p95, p99, min, max per request source (dict/cache/model) |

---

## Scenario 2: Target Scale (Sustained)

**Purpose:** Prove the system meets the target latency and scale bar under realistic sustained load.

| Parameter | Value |
|---|---|
| Users | 500 concurrent (simulated typists) |
| Spawn rate | 50 users/second ramp |
| Duration | 10 minutes sustained after full ramp |
| Request pattern | Each user sends ~0.7 lookups/second (word-boundary debounce) with 1.0-1.5s think time |
| Expected RPS | ~250 to 350 aggregate |
| Metrics to capture | p50, p95, p99, error rate, CPU%, RAM, cache hit ratio |
| Pass criteria | p95 < 50ms server-side (leaves 50ms for network in production) |

---

## Scenario 3: Burst

**Purpose:** Test queueing behavior under sudden load spikes (court session starts, everyone types at once).

| Parameter | Value |
|---|---|
| Users | 1000 (2x target), step function (all at once) |
| Duration | 3 minutes |
| Request pattern | Same as Scenario 2 per user |
| Metrics to capture | p99 during the first 30s (worst case), error rate, recovery time |

---

## Scenario 4: Cache-Off Ablation

**Purpose:** Quantify the value of the dictionary and LRU cache. This number is the core evidence for the architecture decision.

| Parameter | Value |
|---|---|
| Users | 500 concurrent |
| Duration | 5 minutes |
| Server config | Dictionary disabled, LRU disabled, all requests hit model |
| Metrics to capture | Same as Scenario 2 |
| Compare against | Scenario 2 (cache on) |

Expected: latency jumps 10-50x for dictionary-eligible words. This delta is the chart that sells the architecture.

---

## Scenario 5: Pessimistic (Per-Keystroke)

**Purpose:** Find the breaking point. What if the frontend sends per-keystroke instead of per-word?

| Parameter | Value |
|---|---|
| Users | 500 concurrent |
| Duration | 5 minutes |
| Request pattern | ~3 to 5 lookups/second per user (keystroke rate) |
| Expected RPS | ~1,500 to 2,500 aggregate |
| Metrics to capture | Max sustainable throughput, p99, first errors |

---

## Scenario 6: GPU Comparison

**Purpose:** Benchmark the rejected GPU path with identical load, for the cost/latency comparison table in the report.

| Parameter | Value |
|---|---|
| Users | 500 concurrent |
| Duration | 5 minutes |
| Server config | CTranslate2 on GPU (or fairseq on GPU), dictionary ON |
| Metrics to capture | Same as Scenario 2, plus GPU utilization (nvidia-smi) |

Measured result (report section 4.1): CT2 INT8 on the L4 is actually **slower** than one CPU core at batch size 1 (p50 12.3 ms vs 7.4 ms), because kernel-launch and transfer overhead dominate for a tiny char-level model with no batching. Combined with a 344 ms cold start and 5 to 7x the cost, the GPU adds cost with no user-facing gain, so this scenario was not run under load; the microbenchmark already settles it.

---

## Running the Scenarios

```bash
# Scenario 2 example (headless, CSV output):
locust -f locustfile.py \
    --host http://<server-ip>:8000 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 600s \
    --headless \
    --csv results/scenario2_target

# Scenario 4 (cache off, set env var on server side):
DICT_PATH="" LRU_CACHE_SIZE=0 uvicorn server.app:app --port 8000 --workers 4
locust -f locustfile.py \
    --host http://<server-ip>:8000 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 300s \
    --headless \
    --csv results/scenario4_cache_off
```

---

## Important: Test Isolation

The load generator must not contaminate the server's latency numbers. Two ways:

- **Separate machine (ideal):** run Locust on a different VM or your laptop, pointing at the server's IP.
- **Core pinning (used here):** pin the server to 4 cores (`taskset -c 0-3`, 4 workers) and run Locust on separate cores or a separate host so it never competes with the server. Confining the server to 4 vCPUs also makes the numbers approximate an n2-standard-4. See `docs/setup.md` Phase 12 for the exact commands.

Record the hardware specs and the pinning in the results. Both the INT8 and FP32 engines are load-tested (Scenario 2); the FP32 run demonstrates why quantization is required under load.