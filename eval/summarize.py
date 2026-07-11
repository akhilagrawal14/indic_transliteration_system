"""Strip the per-word predictions from an eval result JSON.

`eval/eval.py` writes the full `predictions` map (every input to its candidate
list) so `compare_results.py` can diff two runs. That map embeds the evaluation
dataset, so it is kept local and gitignored. This produces a committable summary:
metrics, timing, config, and a capped sample of top-1 misses for qualitative
evidence, but not the full predictions.

Usage:
    python eval/summarize.py eval/results/ct2_int8.json
    # writes eval/results/ct2_int8.summary.json
"""

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Full eval result JSON")
    parser.add_argument("--output", default="", help="Defaults to <input>.summary.json")
    parser.add_argument("--miss-sample", type=int, default=30,
                        help="How many top-1 misses to retain as evidence")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        full = json.load(handle)

    summary = {k: v for k, v in full.items() if k != "predictions"}
    summary["top1_misses"] = full.get("top1_misses", [])[: args.miss_sample]
    summary["_note"] = (
        "Slimmed for commit. Full per-word predictions live in the "
        "gitignored source file; regenerate with eval/eval.py."
    )

    out = args.output
    if not out:
        base, _ = os.path.splitext(args.input)
        out = f"{base}.summary.json"

    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    full_kb = os.path.getsize(args.input) / 1024
    slim_kb = os.path.getsize(out) / 1024
    print(f"{args.input} ({full_kb:.0f} KB) -> {out} ({slim_kb:.0f} KB)")


if __name__ == "__main__":
    main()
