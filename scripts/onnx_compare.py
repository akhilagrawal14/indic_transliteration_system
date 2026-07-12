"""Benchmark ONNX Runtime vs CTranslate2 on CPU: latency + quality.

Latency on a sample of words (single-word, 1 thread), quality on the full
Dakshina Hindi test set. Reuses the frontier measurement helpers.

Usage:
    python scripts/onnx_compare.py --output eval/results/onnx_comparison.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval import load_references  # noqa: E402
from scripts.frontier import measure_latency, measure_quality  # noqa: E402
from scripts.microbench import load_words  # noqa: E402
from server.engine.ct2_engine import CT2Engine  # noqa: E402
from server.engine.onnx_engine import ONNXEngine  # noqa: E402

TEST_TSV = "eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=TEST_TSV)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--num-words", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--quality-limit", type=int, default=1000,
                        help="Subsample the quality set (0 = full). ONNX has no "
                             "KV cache so the full set is slow.")
    parser.add_argument("--output", default="eval/results/onnx_comparison.json")
    args = parser.parse_args()

    words = load_words(args.data, args.num_words)
    refs = load_references(args.data)
    if args.quality_limit:
        refs = {k: refs[k] for k in sorted(refs)[: args.quality_limit]}

    engines = {
        "ct2_int8": CT2Engine("models/indicxlit/ct2_int8", lang="hi",
                              beam_width=args.beam, topk=args.topk, device="cpu",
                              intra_threads=1, compute_type="int8"),
        "onnx_fp32": ONNXEngine(precision="fp32", beam_width=args.beam,
                                topk=args.topk, intra_threads=1),
        "onnx_int8_dynamic": ONNXEngine(precision="int8", beam_width=args.beam,
                                        topk=args.topk, intra_threads=1),
    }

    rows = []
    for name, eng in engines.items():
        print(f"  benchmarking {name} ...", flush=True)
        lat = measure_latency(eng, words, args.iterations, args.warmup, args.topk)
        qual = measure_quality(eng, refs, args.topk)
        rows.append({"engine": name, **lat, **qual})

    results = {"config": {"beam": args.beam, "topk": args.topk,
                          "num_words": len(words), "eval_inputs": len(refs)},
               "rows": rows}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n| {'engine':<18} | {'p50':>7} | {'p95':>7} | {'p99':>7} | "
          f"{'top-1':>7} | {'top-5':>7} | {'CER':>7} |")
    print("|" + "-" * 20 + "|" + ("-" * 9 + "|") * 6)
    for r in rows:
        print(f"| {r['engine']:<18} | {r['p50_ms']:>7.2f} | {r['p95_ms']:>7.2f} | "
              f"{r['p99_ms']:>7.2f} | {r['top1_accuracy_pct']:>6.2f}% | "
              f"{r['topk_accuracy_pct']:>6.2f}% | {r['cer_top1']:>7.4f} |")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
