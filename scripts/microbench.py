"""Single-word latency microbenchmarks across serving engines.

Reports the standard metrics for each engine: latency percentiles
(p50/p95/p99), throughput, cold-start time, and process memory.

Usage:
    python scripts/microbench.py --output eval/results/microbench.json
    python scripts/microbench.py --print-table eval/results/microbench.json
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional

import psutil

# Repo root on sys.path so `server.compat` imports when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_DIR = "models/indicxlit/ct2_int8"
DEFAULT_WORDS_TSV = "eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
DEFAULT_DICT_TSV = "eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"


def load_words(path: str, n: int, seed: int = 42) -> List[str]:
    """Sample n unique romanized words from a Dakshina lexicon TSV.

    Format is `native \\t romanized \\t count`.
    """
    romanized = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                romanized.add(parts[1])
    words = sorted(romanized)
    random.Random(seed).shuffle(words)
    return words[:n]


def build_dictionary(path: str) -> Dict[str, List[str]]:
    """Build romanized -> ranked native candidates from a Dakshina lexicon.

    Candidates are ordered by the corpus count column, descending.
    """
    scored: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            native, roman, count = parts[0], parts[1], parts[2]
            if not native or not roman:
                continue
            scored.setdefault(roman, {})
            scored[roman][native] = scored[roman].get(native, 0) + int(count or 0)
    return {
        roman: [w for w, _ in sorted(cands.items(), key=lambda kv: -kv[1])]
        for roman, cands in scored.items()
    }


def percentiles(samples_ms: List[float]) -> Dict[str, float]:
    """Return standard latency metrics in milliseconds."""
    ordered = sorted(samples_ms)

    def pct(p: float) -> float:
        # Nearest-rank percentile; exact and dependency-free.
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
        return ordered[idx]

    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "throughput_qps": round(1000.0 / statistics.fmean(ordered), 1),
    }


def measure(fn: Callable[[str], object], words: List[str], iterations: int,
            warmup: int) -> Dict[str, float]:
    """Time fn once per word, sequentially, after a warmup."""
    for i in range(warmup):
        fn(words[i % len(words)])

    samples: List[float] = []
    for i in range(iterations):
        word = words[i % len(words)]
        start = time.perf_counter()
        fn(word)
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentiles(samples)


def rss_mb() -> float:
    """Resident set size of this process, in MiB."""
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def bench_ct2(model_dir: str, device: str, intra_threads: int, words: List[str],
              iterations: int, warmup: int, beam: int, topk: int,
              lang: str) -> Dict[str, object]:
    """Benchmark the CTranslate2 INT8 engine."""
    import ctranslate2

    rss_before = rss_mb()
    start = time.perf_counter()
    translator = ctranslate2.Translator(
        model_dir, device=device, compute_type="int8", intra_threads=intra_threads
    )
    cold_start_ms = (time.perf_counter() - start) * 1000.0

    def run(word: str) -> List[str]:
        src = [f"__{lang}__"] + list(word.lower())
        result = translator.translate_batch(
            [src], beam_size=beam, num_hypotheses=topk
        )[0]
        return ["".join(h) for h in result.hypotheses]

    stats = measure(run, words, iterations, warmup)
    stats["cold_start_ms"] = round(cold_start_ms, 1)
    stats["rss_delta_mb"] = round(rss_mb() - rss_before, 1)
    return stats


def bench_fairseq(words: List[str], iterations: int, warmup: int, beam: int,
                  topk: int, lang: str) -> Dict[str, object]:
    """Benchmark the stock fairseq XlitEngine (FP32, CPU only)."""
    from server.compat import stub_urduhack

    stub_urduhack()
    from ai4bharat.transliteration import XlitEngine

    rss_before = rss_mb()
    start = time.perf_counter()
    engine = XlitEngine(lang, beam_width=beam, rescore=False)
    cold_start_ms = (time.perf_counter() - start) * 1000.0

    def run(word: str) -> List[str]:
        return engine.translit_word(word, lang_code=lang, topk=topk)

    stats = measure(run, words, iterations, warmup)
    stats["cold_start_ms"] = round(cold_start_ms, 1)
    stats["rss_delta_mb"] = round(rss_mb() - rss_before, 1)
    return stats


def bench_dictionary(dict_tsv: str, words: List[str], iterations: int,
                     warmup: int) -> Dict[str, object]:
    """Benchmark the precomputed dictionary lookup path."""
    rss_before = rss_mb()
    start = time.perf_counter()
    table = build_dictionary(dict_tsv)
    cold_start_ms = (time.perf_counter() - start) * 1000.0

    def run(word: str) -> Optional[List[str]]:
        return table.get(word.lower())

    stats = measure(run, words, iterations, warmup)
    stats["cold_start_ms"] = round(cold_start_ms, 1)
    stats["rss_delta_mb"] = round(rss_mb() - rss_before, 1)
    stats["entries"] = len(table)
    return stats


def print_table(results: Dict[str, object]) -> None:
    """Print the benchmark results as a markdown table."""
    header = (
        f"| {'Engine':<24} | {'Device':<8} | {'p50':>8} | {'p95':>8} | "
        f"{'p99':>8} | {'mean':>8} | {'QPS':>8} | {'cold start':>11} |"
    )
    print(header)
    print("|" + "-" * 26 + "|" + "-" * 10 + "|" + ("-" * 10 + "|") * 5 + "-" * 13 + "|")
    for name, r in results["engines"].items():
        if "error" in r:
            print(f"| {name:<24} | {'-':<8} | {'SKIPPED: ' + r['error']:<60} |")
            continue
        print(
            f"| {name:<24} | {r['device']:<8} | {r['p50_ms']:>8.3f} | "
            f"{r['p95_ms']:>8.3f} | {r['p99_ms']:>8.3f} | {r['mean_ms']:>8.3f} | "
            f"{r['throughput_qps']:>8.1f} | {r['cold_start_ms']:>9.1f}ms |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-table", metavar="JSON",
                        help="Print a saved results JSON as a table and exit")
    parser.add_argument("--output", default="eval/results/microbench.json")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--words-tsv", default=DEFAULT_WORDS_TSV)
    parser.add_argument("--dict-tsv", default=DEFAULT_DICT_TSV)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-words", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--fairseq-iterations", type=int, default=150,
                        help="Fewer iterations: the FP32 baseline is slow")
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.print_table:
        with open(args.print_table, encoding="utf-8") as handle:
            print_table(json.load(handle))
        return

    words = load_words(args.words_tsv, args.num_words)
    print(f"Benchmarking on {len(words)} unique romanized words "
          f"(beam={args.beam}, topk={args.topk})\n")

    results: Dict[str, object] = {
        "config": {
            "beam_width": args.beam,
            "topk": args.topk,
            "num_words": len(words),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "engines": {},
    }

    print("[1/5] dictionary lookup (CPU)...")
    r = bench_dictionary(args.dict_tsv, words, args.iterations, args.warmup)
    r["device"] = "cpu"
    results["engines"]["dictionary"] = r

    print("[2/5] ct2 int8, CPU, 1 thread...")
    r = bench_ct2(args.model_dir, "cpu", 1, words, args.iterations, args.warmup,
                  args.beam, args.topk, args.lang)
    r["device"] = "cpu x1"
    results["engines"]["ct2_int8_1thread"] = r

    print("[3/5] ct2 int8, CPU, all threads...")
    r = bench_ct2(args.model_dir, "cpu", 0, words, args.iterations, args.warmup,
                  args.beam, args.topk, args.lang)
    r["device"] = "cpu xN"
    results["engines"]["ct2_int8_allthreads"] = r

    print("[4/5] ct2 int8, GPU (L4)...")
    try:
        r = bench_ct2(args.model_dir, "cuda", 0, words, args.iterations,
                      args.warmup, args.beam, args.topk, args.lang)
        r["device"] = "L4"
        results["engines"]["ct2_int8_gpu"] = r
    except Exception as exc:  # noqa: BLE001
        results["engines"]["ct2_int8_gpu"] = {"error": str(exc)[:80]}

    print("[5/5] fairseq FP32 (CPU only, stock XlitEngine)...")
    r = bench_fairseq(words, args.fairseq_iterations, args.warmup, args.beam,
                      args.topk, args.lang)
    r["device"] = "cpu"
    results["engines"]["fairseq_fp32"] = r

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nWrote {args.output}\n")
    print_table(results)


if __name__ == "__main__":
    main()
