"""Build a small client-side dictionary for the demo's offline path.

Takes the top-N most frequent romanized inputs (ranked by the Dakshina natural
corpus token frequency) from the full precomputed dictionary and writes a
single-digit-MB JSON the browser bundles. When the API is unreachable (flaky
courtroom connectivity), the demo serves head-word suggestions from this file.

Usage:
    python scripts/build_client_dict.py --top-n 10000
"""

import argparse
import json
import os
import re
from collections import Counter

FULL_DICT = "server/data/dictionary_hi.json"
CORPUS = "eval/data/dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.dev.roman.txt"
OUT = "demo/public/client_dict_hi.json"
TOKEN_RE = re.compile(r"[a-z]+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dict", default=FULL_DICT)
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--output", default=OUT)
    args = parser.parse_args()

    with open(args.full_dict, encoding="utf-8") as f:
        full = json.load(f)

    # Rank dictionary entries by corpus frequency; fall back to alphabetical for
    # entries not seen in the corpus so the file is deterministic.
    with open(args.corpus, encoding="utf-8") as f:
        freq = Counter(TOKEN_RE.findall(f.read().lower()))

    ranked = sorted(full.keys(), key=lambda w: (-freq.get(w, 0), w))
    top = ranked[: args.top_n]
    client = {w: full[w] for w in top}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(client, f, ensure_ascii=False, separators=(",", ":"))

    covered = sum(freq.get(w, 0) for w in top)
    total = sum(freq.values()) or 1
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {len(client):,} entries to {args.output} ({size_mb:.1f} MB)")
    print(f"Corpus token coverage of this subset: {100.0 * covered / total:.1f}%")


if __name__ == "__main__":
    main()
