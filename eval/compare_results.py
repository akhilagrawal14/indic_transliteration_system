"""Diff two eval result JSONs to quantify a quality delta.

Reports the metric deltas and, more usefully, categorizes every input where the
two engines disagree: did the candidate engine get worse, better, or merely
return a different but equally acceptable ranking?

Usage:
    python eval/compare_results.py \\
        --baseline eval/results/baseline_fp32.json \\
        --candidate eval/results/ct2_int8.json \\
        --output eval/results/quality_delta.json
"""

import argparse
import json
import os
from typing import Dict, List


def load(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--show", type=int, default=10,
                        help="How many example disagreements to print")
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    bm, cm = base["metrics"], cand["metrics"]
    bp: Dict[str, List[str]] = base["predictions"]
    cp: Dict[str, List[str]] = cand["predictions"]

    shared = sorted(set(bp) & set(cp))
    top1_same = sum(1 for w in shared if bp[w][:1] == cp[w][:1])
    top5_same = sum(1 for w in shared if bp[w] == cp[w])

    # Where the top-1 changed, did accuracy move?
    regressions: List[Dict[str, object]] = []
    for w in shared:
        if bp[w][:1] == cp[w][:1]:
            continue
        entry = {"input": w, "baseline": bp[w], "candidate": cp[w]}
        # We do not have the reference set here, so classify by miss lists.
        regressions.append(entry)

    base_miss = {m["input"] for m in base.get("top1_misses", [])}
    cand_miss = {m["input"] for m in cand.get("top1_misses", [])}
    newly_wrong = sorted(cand_miss - base_miss)
    newly_right = sorted(base_miss - cand_miss)

    delta = {
        "baseline_engine": base["engine"],
        "candidate_engine": cand["engine"],
        "num_inputs": len(shared),
        "metrics": {
            "top1_accuracy_pct": {
                "baseline": bm["top1_accuracy_pct"],
                "candidate": cm["top1_accuracy_pct"],
                "delta": round(cm["top1_accuracy_pct"] - bm["top1_accuracy_pct"], 2),
            },
            "top5_accuracy_pct": {
                "baseline": bm["top5_accuracy_pct"],
                "candidate": cm["top5_accuracy_pct"],
                "delta": round(cm["top5_accuracy_pct"] - bm["top5_accuracy_pct"], 2),
            },
            "cer_top1": {
                "baseline": bm["cer_top1"],
                "candidate": cm["cer_top1"],
                "delta": round(cm["cer_top1"] - bm["cer_top1"], 4),
            },
        },
        "agreement": {
            "identical_top1_pct": round(100.0 * top1_same / len(shared), 2),
            "identical_full_ranking_pct": round(100.0 * top5_same / len(shared), 2),
            "top1_changed": len(shared) - top1_same,
        },
        "top1_regressions_count": len(newly_wrong),
        "top1_improvements_count": len(newly_right),
        "examples": {
            "top1_changed": regressions[: args.show],
            "newly_wrong": newly_wrong[: args.show],
            "newly_right": newly_right[: args.show],
        },
    }

    m = delta["metrics"]
    print(f"{'metric':<22} {'baseline':>10} {'candidate':>10} {'delta':>10}")
    print("-" * 56)
    for key in ("top1_accuracy_pct", "top5_accuracy_pct", "cer_top1"):
        print(f"{key:<22} {m[key]['baseline']:>10} {m[key]['candidate']:>10} "
              f"{m[key]['delta']:>+10}")

    a = delta["agreement"]
    print(f"\nidentical top-1 ranking : {a['identical_top1_pct']}%")
    print(f"identical full top-5    : {a['identical_full_ranking_pct']}%")
    print(f"top-1 changed           : {a['top1_changed']} inputs")
    print(f"  became wrong          : {delta['top1_regressions_count']}")
    print(f"  became right          : {delta['top1_improvements_count']}")

    if delta["examples"]["top1_changed"]:
        print(f"\nExample top-1 disagreements (first {args.show}):")
        for e in delta["examples"]["top1_changed"]:
            print(f"  {e['input']:<14} base={e['baseline'][:3]}  cand={e['candidate'][:3]}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(delta, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
