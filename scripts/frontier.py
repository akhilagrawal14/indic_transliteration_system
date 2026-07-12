"""Measure the latency/quality frontier for the CTranslate2 CPU engine.

Two sweeps, both on this box (Cascade Lake CPU, 1 intra-op thread, to match the
serving config):

  --sweep precision : compute_type in {int8, int8_float32, int16, float32}
  --sweep beam      : beam width in {1, 3, 5, 8} at a fixed compute_type

For each config it reports single-word latency percentiles (p50/p95/p99) and
quality (top-1, top-5, CER) on the Dakshina Hindi test set, so a precision or
beam choice can be justified with data rather than asserted.

Usage:
    python scripts/frontier.py --sweep precision --output eval/results/precision_frontier.json
    python scripts/frontier.py --sweep beam --output eval/results/beam_frontier.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval import best_cer, load_references  # noqa: E402
from scripts.microbench import load_words, percentiles  # noqa: E402
from server.engine.ct2_engine import CT2Engine  # noqa: E402

DEFAULT_MODEL_DIR = "models/indicxlit/ct2_int8"
TEST_TSV = "eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"

PRECISIONS = ["int8", "int8_float32", "int16", "float32"]
BEAMS = [1, 3, 5, 8]


def measure_latency(engine: CT2Engine, words: List[str], iterations: int,
                    warmup: int, topk: int) -> Dict[str, float]:
    """Single-word latency percentiles, one word per call."""
    for i in range(warmup):
        engine.transliterate(words[i % len(words)], topk=topk)
    samples: List[float] = []
    for i in range(iterations):
        start = time.perf_counter()
        engine.transliterate(words[i % len(words)], topk=topk)
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentiles(samples)


def measure_quality(engine: CT2Engine, refs: Dict[str, set], topk: int,
                    batch_size: int = 64) -> Dict[str, float]:
    """Top-1 / top-5 accuracy and CER on the reference set."""
    inputs = sorted(refs)
    top1 = top5 = 0
    cer_total = 0.0
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i: i + batch_size]
        for word, cands in zip(batch, engine.transliterate_batch(batch, topk)):
            acceptable = refs[word]
            top1 += bool(cands) and cands[0] in acceptable
            top5 += any(c in acceptable for c in cands[:topk])
            cer_total += best_cer(cands[0], acceptable) if cands else 1.0
    n = len(inputs)
    return {
        "top1_accuracy_pct": round(100.0 * top1 / n, 2),
        "topk_accuracy_pct": round(100.0 * top5 / n, 2),
        "cer_top1": round(cer_total / n, 4),
    }


def run_config(model_dir: str, compute_type: str, beam: int, topk: int,
               words: List[str], refs: Dict[str, set], iterations: int,
               warmup: int) -> Dict[str, object]:
    """Build an engine at (compute_type, beam) and measure latency + quality.

    Beam width caps the number of candidates, so the evaluated k is min(beam,
    topk). This is the point of the beam sweep: beam < 5 cannot serve a top-5.
    """
    eval_k = min(beam, topk)
    engine = CT2Engine(model_dir, lang="hi", beam_width=beam, topk=eval_k,
                       device="cpu", intra_threads=1, compute_type=compute_type)
    latency = measure_latency(engine, words, iterations, warmup, eval_k)
    quality = measure_quality(engine, refs, eval_k)
    return {"compute_type": compute_type, "beam": beam, "eval_k": eval_k,
            **latency, **quality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["precision", "beam"], required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data", default=TEST_TSV)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--fixed-compute-type", default="int8",
                        help="compute_type used during the beam sweep")
    parser.add_argument("--fixed-beam", type=int, default=5,
                        help="beam used during the precision sweep")
    parser.add_argument("--num-words", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    words = load_words(args.data, args.num_words)
    refs = load_references(args.data)

    if args.sweep == "precision":
        configs = [(ct, args.fixed_beam) for ct in PRECISIONS]
    else:
        configs = [(args.fixed_compute_type, b) for b in BEAMS]

    print(f"sweep={args.sweep} words={len(words)} eval_inputs={len(refs):,} "
          f"topk={args.topk}\n")

    rows: List[Dict[str, object]] = []
    for compute_type, beam in configs:
        print(f"  running compute_type={compute_type} beam={beam} ...", flush=True)
        rows.append(run_config(args.model_dir, compute_type, beam, args.topk,
                               words, refs, args.iterations, args.warmup))

    results = {
        "sweep": args.sweep,
        "config": {"num_words": len(words), "iterations": args.iterations,
                   "topk": args.topk, "eval_inputs": len(refs)},
        "rows": rows,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    label = "compute_type" if args.sweep == "precision" else "beam"
    print(f"\n| {label:<14} | {'p50':>7} | {'p95':>7} | {'p99':>7} | "
          f"{'top-1':>7} | {'top-k':>7} | {'k':>2} | {'CER':>7} |")
    print("|" + "-" * 16 + "|" + ("-" * 9 + "|") * 4 + "----|" + "-" * 9 + "|")
    for r in rows:
        key = r["compute_type"] if args.sweep == "precision" else r["beam"]
        print(f"| {str(key):<14} | {r['p50_ms']:>7.3f} | {r['p95_ms']:>7.3f} | "
              f"{r['p99_ms']:>7.3f} | {r['top1_accuracy_pct']:>6.2f}% | "
              f"{r['topk_accuracy_pct']:>6.2f}% | {r['eval_k']:>2} | "
              f"{r['cer_top1']:>7.4f} |")
    if args.output:
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
