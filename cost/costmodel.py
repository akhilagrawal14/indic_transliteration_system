"""Cost model: monthly serving cost vs traffic, CPU (chosen) vs always-on GPU.

Grounded in measured numbers from this project:
  - per-core model throughput (from scripts/microbench.py)
  - dictionary cache-hit ratio (from the load test)
and real GCP on-demand instance prices (asia-south1, provided).

Outputs $/month, $/1k users, and $/1M lookups at 1x / 3x / 10x scale, plus a
tally of the experiment/dev cost incurred so far. Re-run whenever new
measurements land so the numbers track what has actually been done.

Usage:
    python cost/costmodel.py --output cost/results/cost_comparison.json
"""

import argparse
import json
import math
import os
from typing import Dict, List

# --- GCP on-demand monthly prices (USD), user-provided ---
# 4 vCPU / 16 GB / 30 GB disk for CPU; 1x L4 for GPU.
PRICES = {
    "n2-standard-4": 144.79,   # recommended CPU serving (AVX-512 VNNI)
    "c3-standard-4": 149.57,   # alt CPU serving (Sapphire Rapids)
    "g2-standard-4": 518.99,   # rejected always-on GPU serving (1x L4)
    "g2-standard-8": 626.15,   # GPU precompute box (not needed; CPU suffices)
}

# --- Measured performance (this project) ---
# CT2 INT8, single word, 1 CPU core: p50 7.39 ms -> ~135 model-RPS/core ceiling.
# Derate to a sustainable rate that keeps p95 low under real concurrency.
MODEL_RPS_PER_CORE = 100.0
CORES_PER_CPU_INSTANCE = 4
# Measured aggregate ceiling for a 4-core box (load test): ~950 RPS at saturation,
# ~800 RPS while staying inside the latency budget. This HTTP-throughput ceiling,
# not the model path, is what actually sizes the fleet.
AGG_RPS_PER_CPU_INSTANCE = 800.0
GPU_MODEL_RPS = 80.0            # CT2 INT8 on L4, batch=1 (measured ~76 QPS)


def workload(dau: int, session_hours: float, active_fraction: float,
             req_per_active_sec: float, peak_concurrency_fraction: float
             ) -> Dict[str, float]:
    """Derive daily volume and peak RPS from one consistent typing model.

    Every quantity flows from the same parameters, so daily volume and peak RPS
    can't disagree (the earlier version mixed a 1500/hour figure with a 0.7 rps
    figure). `lookups_per_user_hour` is now a derived output, not an input.
    """
    lookups_per_user_hour = 3600 * active_fraction * req_per_active_sec
    lookups_per_user_day = session_hours * lookups_per_user_hour
    daily_lookups = dau * lookups_per_user_day

    peak_concurrent = dau * peak_concurrency_fraction
    # Of the concurrent users, active_fraction are typing at req_per_active_sec.
    peak_rps = peak_concurrent * active_fraction * req_per_active_sec
    return {
        "lookups_per_user_hour": round(lookups_per_user_hour),
        "daily_lookups": daily_lookups,
        "monthly_lookups": daily_lookups * 30,
        "peak_concurrent": peak_concurrent,
        "peak_rps": peak_rps,
    }


def cpu_plan(peak_rps: float, hit_ratio: float, min_instances: int) -> Dict[str, float]:
    """CPU instances needed.

    Two constraints: the model path (only cache misses) and the measured aggregate
    HTTP-throughput ceiling per box. The aggregate ceiling binds first at scale.
    """
    model_rps = peak_rps * (1.0 - hit_ratio)
    model_capacity = MODEL_RPS_PER_CORE * CORES_PER_CPU_INSTANCE
    by_model = math.ceil(model_rps / model_capacity)
    by_aggregate = math.ceil(peak_rps / AGG_RPS_PER_CPU_INSTANCE)
    needed = max(min_instances, by_model, by_aggregate)
    return {"model_rps": model_rps, "instances": needed,
            "bound_by": "aggregate" if by_aggregate >= by_model else "model"}


def gpu_plan(peak_rps: float, hit_ratio: float, min_instances: int) -> Dict[str, float]:
    """Always-on GPU sizing for the same load (the rejected option)."""
    model_rps = peak_rps * (1.0 - hit_ratio)
    needed = max(min_instances, math.ceil(model_rps / GPU_MODEL_RPS))
    return {"model_rps": model_rps, "instances": needed}


def scale_row(mult: int, base: Dict, hit_ratio: float, cpu_price: float,
              gpu_price: float, min_instances: int) -> Dict:
    """Cost at a given traffic multiple."""
    dau = base["dau"] * mult
    wl = workload(dau, base["session_hours"], base["active_fraction"],
                  base["req_per_active_sec"], base["peak_concurrency_fraction"])
    cpu = cpu_plan(wl["peak_rps"], hit_ratio, min_instances)
    gpu = gpu_plan(wl["peak_rps"], hit_ratio, min_instances)
    cpu_cost = cpu["instances"] * cpu_price
    gpu_cost = gpu["instances"] * gpu_price
    monthly_lookups_m = wl["monthly_lookups"] / 1e6
    return {
        "scale": f"{mult}x",
        "dau": dau,
        "peak_rps": round(wl["peak_rps"]),
        "model_rps": round(cpu["model_rps"]),
        "cpu_instances": cpu["instances"],
        "cpu_usd_month": round(cpu_cost, 2),
        "gpu_instances": gpu["instances"],
        "gpu_usd_month": round(gpu_cost, 2),
        "cpu_usd_per_1k_users_month": round(cpu_cost / (dau / 1000), 2),
        "cpu_usd_per_1M_lookups": round(cpu_cost / monthly_lookups_m, 2),
        "gpu_usd_per_1M_lookups": round(gpu_cost / monthly_lookups_m, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dau", type=int, default=5000)
    parser.add_argument("--session-hours", type=float, default=3.0,
                        help="Court-session hours per user per day")
    parser.add_argument("--active-fraction", type=float, default=0.6,
                        help="Fraction of session actively typing (duty cycle)")
    parser.add_argument("--req-per-active-sec", type=float, default=0.7,
                        help="Lookups/sec while actively typing")
    parser.add_argument("--peak-concurrency-fraction", type=float, default=0.10,
                        help="Fraction of DAU concurrent at peak")
    parser.add_argument("--hit-ratio", type=float, default=0.92,
                        help="Dictionary cache-hit ratio (measured in load test)")
    parser.add_argument("--cpu-instance", default="n2-standard-4")
    parser.add_argument("--gpu-instance", default="g2-standard-4")
    parser.add_argument("--min-instances", type=int, default=2,
                        help="Floor for HA (2 instances behind a load balancer)")
    parser.add_argument("--output", default="cost/results/cost_comparison.json")
    args = parser.parse_args()

    base = {
        "dau": args.dau,
        "session_hours": args.session_hours,
        "active_fraction": args.active_fraction,
        "req_per_active_sec": args.req_per_active_sec,
        "peak_concurrency_fraction": args.peak_concurrency_fraction,
    }
    cpu_price = PRICES[args.cpu_instance]
    gpu_price = PRICES[args.gpu_instance]
    derived = workload(args.dau, args.session_hours, args.active_fraction,
                       args.req_per_active_sec, args.peak_concurrency_fraction)

    rows: List[Dict] = [
        scale_row(m, base, args.hit_ratio, cpu_price, gpu_price, args.min_instances)
        for m in (1, 3, 10)
    ]

    # One-time GPU evaluation cost: the rejected GPU path was benchmarked on a
    # GPU machine (1x L4) before settling on CPU-only serving. Rough tally.
    gpu_test_hourly = PRICES["g2-standard-4"] / 730.0   # 1x L4 machine
    gpu_test_hours = 8.0
    experiment_cost = round(gpu_test_hourly * gpu_test_hours, 2)

    results = {
        "assumptions": {
            "dau": args.dau,
            "session_hours": args.session_hours,
            "active_fraction": args.active_fraction,
            "req_per_active_sec": args.req_per_active_sec,
            "peak_concurrency_fraction": args.peak_concurrency_fraction,
            "derived_lookups_per_user_hour": derived["lookups_per_user_hour"],
            "derived_peak_rps_1x": round(derived["peak_rps"]),
            "cache_hit_ratio": args.hit_ratio,
            "model_rps_per_core": MODEL_RPS_PER_CORE,
            "agg_rps_per_cpu_instance": AGG_RPS_PER_CPU_INSTANCE,
            "cpu_instance": f"{args.cpu_instance} @ ${cpu_price}/mo",
            "gpu_instance": f"{args.gpu_instance} @ ${gpu_price}/mo",
            "ha_min_instances": args.min_instances,
        },
        "excluded_costs": [
            "HTTPS load balancer", "CDN request/egress", "cross-zone/region egress",
            "monitoring/log ingestion", "artifact/registry storage",
            "sustained-use/committed-use discounts", "engineering/ops overhead",
        ],
        "scale": rows,
        "gpu_evaluation_cost": {
            "machine": "GPU machine, 1x L4",
            "approx_hours": gpu_test_hours,
            "approx_usd": experiment_cost,
            "note": "One-time cost to benchmark and reject the GPU path; serving is CPU-only.",
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"CPU: {args.cpu_instance} @ ${cpu_price}/mo | "
          f"GPU: {args.gpu_instance} @ ${gpu_price}/mo | "
          f"hit ratio {args.hit_ratio:.1%}\n")
    print(f"| {'scale':<5} | {'DAU':>7} | {'peak RPS':>8} | {'model RPS':>9} | "
          f"{'CPU $/mo':>9} | {'GPU $/mo':>9} | {'CPU $/1M':>9} |")
    print("|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 11 + "|"
          + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|")
    for r in rows:
        print(f"| {r['scale']:<5} | {r['dau']:>7,} | {r['peak_rps']:>8} | "
              f"{r['model_rps']:>9} | {r['cpu_usd_month']:>9,.0f} | "
              f"{r['gpu_usd_month']:>9,.0f} | {r['cpu_usd_per_1M_lookups']:>9.2f} |")
    print(f"\nOne-time GPU evaluation cost: ~${experiment_cost} "
          f"({gpu_test_hours} h on a 1x-L4 GPU machine; serving is CPU-only)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
