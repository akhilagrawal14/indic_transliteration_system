"""Quality evaluation on the Dakshina Hindi romanization lexicon.

Reports top-1 accuracy, top-5 accuracy, and character error rate, so a quantized
engine can be compared against the FP32 baseline.

The Dakshina lexicon is stored native-first (`native \\t romanized \\t count`) and
is many-to-one: several native spellings can share one romanization (for example
`ankhon` and `anakon` both romanize `अंकों`). We therefore invert it to
`romanized -> {acceptable native forms}` and score a prediction correct if it
matches any acceptable form. Scoring against a single reference would understate
accuracy.

Usage:
    python eval/eval.py --engine ct2 --lang hi --topk 5 --output eval/results/ct2_int8.json
    python eval/eval.py --engine fairseq --lang hi --topk 5 --output eval/results/baseline_fp32.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.engine.base import TransliterationEngine  # noqa: E402

DEFAULT_DATA = "eval/data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
DEFAULT_MODEL_DIR = "models/indicxlit/ct2_int8"


def load_references(path: str) -> Dict[str, Set[str]]:
    """Invert a Dakshina lexicon into romanized -> set of acceptable natives."""
    refs: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            refs.setdefault(parts[1], set()).add(parts[0])
    return refs


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def best_cer(prediction: str, references: Set[str]) -> float:
    """Character error rate of `prediction` against its closest reference."""
    return min(
        edit_distance(prediction, ref) / max(len(ref), 1) for ref in references
    )


def build_engine(name: str, lang: str, beam: int, topk: int,
                 model_dir: str, device: str) -> TransliterationEngine:
    """Construct the requested engine."""
    if name == "ct2":
        from server.engine.ct2_engine import CT2Engine
        return CT2Engine(model_dir, lang=lang, beam_width=beam, topk=topk,
                         device=device)
    if name == "fairseq":
        from server.engine.fairseq_engine import FairseqEngine
        return FairseqEngine(lang=lang, beam_width=beam, topk=topk)
    raise ValueError(f"unknown engine: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["ct2", "fairseq"])
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means all inputs")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    refs = load_references(args.data)
    inputs = sorted(refs)
    if args.limit:
        inputs = inputs[: args.limit]

    print(f"engine={args.engine} inputs={len(inputs):,} beam={args.beam} topk={args.topk}")
    engine = build_engine(args.engine, args.lang, args.beam, args.topk,
                          args.model_dir, args.device)

    predictions: Dict[str, List[str]] = {}
    start = time.perf_counter()
    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i: i + args.batch_size]
        for word, cands in zip(batch, engine.transliterate_batch(batch, args.topk)):
            predictions[word] = cands
        if i % (args.batch_size * 20) == 0:
            print(f"  {i:,}/{len(inputs):,}", flush=True)
    elapsed = time.perf_counter() - start

    top1 = top5 = 0
    cer_total = 0.0
    misses: List[Dict[str, object]] = []
    for word in inputs:
        cands = predictions[word]
        acceptable = refs[word]
        hit1 = bool(cands) and cands[0] in acceptable
        hit5 = any(c in acceptable for c in cands[: args.topk])
        top1 += hit1
        top5 += hit5
        cer_total += best_cer(cands[0], acceptable) if cands else 1.0
        if not hit1:
            misses.append({
                "input": word,
                "expected": sorted(acceptable),
                "predicted": cands,
                "top5_hit": hit5,
            })

    n = len(inputs)
    results = {
        "engine": args.engine,
        "device": args.device,
        "lang": args.lang,
        "beam_width": args.beam,
        "topk": args.topk,
        "data": args.data,
        "num_inputs": n,
        "metrics": {
            "top1_accuracy_pct": round(100.0 * top1 / n, 2),
            "top5_accuracy_pct": round(100.0 * top5 / n, 2),
            "cer_top1": round(cer_total / n, 4),
        },
        "timing": {
            "total_seconds": round(elapsed, 1),
            "ms_per_word": round(elapsed / n * 1000, 3),
        },
        "predictions": predictions,
        "top1_misses": misses[:200],
    }

    print(f"\ntop-1 accuracy : {results['metrics']['top1_accuracy_pct']:.2f}%")
    print(f"top-5 accuracy : {results['metrics']['top5_accuracy_pct']:.2f}%")
    print(f"CER (top-1)    : {results['metrics']['cer_top1']:.4f}")
    print(f"elapsed        : {elapsed:.1f}s ({results['timing']['ms_per_word']:.2f} ms/word)")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
