"""Measure how much of the lookup volume a precomputed dictionary would serve.

The architecture assumes courtroom vocabulary is Zipfian, so a top-N precomputed
dictionary covers most real lookups. This script tests that assumption two ways:

1. Lexicon view (`--mode lexicon`): uses the Dakshina `translit.sampled` lexicon.
   This is the WRONG instrument and is kept only to document why. The lexicon is
   a sampled word list balanced across the frequency spectrum, and its train/test
   splits are disjoint by construction, so held-out hit rate is meaningless as a
   cache-hit proxy.

2. Corpus view (`--mode corpus`, the default): uses Dakshina's natural romanized
   sentence data, which has a real token frequency distribution. Dictionary is
   built from the dev split plus the lexicon; traffic is the test split, counted
   with repeats (volume weighted). This is the honest cache-hit estimate.

Usage:
    python scripts/zipf_coverage.py --output eval/results/zipf_coverage.json
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Set

BASE = "eval/data/dakshina_dataset_v1.0/hi"
LEXICON_TRAIN = f"{BASE}/lexicons/hi.translit.sampled.train.tsv"
LEXICON_TEST = f"{BASE}/lexicons/hi.translit.sampled.test.tsv"
CORPUS_DEV = f"{BASE}/romanized/hi.romanized.rejoined.dev.roman.txt"
CORPUS_TEST = f"{BASE}/romanized/hi.romanized.rejoined.test.roman.txt"
# The bundled browser offline dictionary (a subset of the server dictionary). Its
# coverage on natural *test* traffic is the honest offline hit rate, as opposed to
# self-coverage measured on the corpus it was selected from.
CLIENT_DICT = "demo/public/client_dict_hi.json"

TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(path: str) -> List[str]:
    """Lowercase word tokens from a romanized text file, punctuation stripped."""
    with open(path, encoding="utf-8") as handle:
        return TOKEN_RE.findall(handle.read().lower())


def lexicon_inputs(path: str) -> Set[str]:
    """Unique romanized inputs from a Dakshina lexicon TSV (native, roman, count)."""
    inputs: Set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                inputs.add(parts[1].lower())
    return inputs


def zipf_curve(counts: Counter, cutoffs: List[int]) -> List[Dict[str, float]]:
    """Fraction of total token volume covered by the top-N most frequent types."""
    ordered = [c for _, c in counts.most_common()]
    total = sum(ordered)
    rows = []
    for n in cutoffs:
        if n > len(ordered):
            continue
        rows.append({
            "top_n": n,
            "coverage_pct": round(100.0 * sum(ordered[:n]) / total, 2),
        })
    return rows


def hit_rate(traffic: Counter, dictionary: Set[str]) -> Dict[str, float]:
    """Volume-weighted and unique-type hit rate of a dictionary against traffic."""
    total_volume = sum(traffic.values())
    hit_volume = sum(c for tok, c in traffic.items() if tok in dictionary)
    hit_types = sum(1 for tok in traffic if tok in dictionary)
    return {
        "volume_hit_rate_pct": round(100.0 * hit_volume / total_volume, 2),
        "unique_hit_rate_pct": round(100.0 * hit_types / len(traffic), 2),
        "dict_entries": len(dictionary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval/results/zipf_coverage.json")
    args = parser.parse_args()

    # --- Corpus view: the honest measurement ---
    dev_tokens = tokenize(CORPUS_DEV)
    test_tokens = tokenize(CORPUS_TEST)
    dev_counts = Counter(dev_tokens)
    test_counts = Counter(test_tokens)

    cutoffs = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    self_curve = zipf_curve(test_counts, cutoffs)

    lex_train = lexicon_inputs(LEXICON_TRAIN)
    dev_vocab = set(dev_counts)

    scenarios = {
        "lexicon_only": hit_rate(test_counts, lex_train),
        "corpus_dev_only": hit_rate(test_counts, dev_vocab),
        "lexicon_plus_corpus_dev": hit_rate(test_counts, lex_train | dev_vocab),
    }

    # Browser offline dictionary, measured on held-out test traffic (not the
    # corpus it was selected from), so this is true offline coverage. Skipped if
    # the generated artifact is not present (it is gitignored; run
    # scripts/build_client_dict.py first).
    if os.path.exists(CLIENT_DICT):
        with open(CLIENT_DICT, encoding="utf-8") as handle:
            client_keys = {k.lower() for k in json.load(handle)}
        scenarios["client_dict_offline"] = hit_rate(test_counts, client_keys)

    # --- Lexicon view: documented as the wrong instrument ---
    lex_test = lexicon_inputs(LEXICON_TEST)
    lexicon_view = {
        "train_unique": len(lex_train),
        "test_unique": len(lex_test),
        "held_out_unique_hit_rate_pct": round(
            100.0 * len(lex_train & lex_test) / len(lex_test), 2
        ),
        "note": "Splits are disjoint by construction; not a cache-hit proxy.",
    }

    results = {
        "corpus": {
            "dev_tokens": len(dev_tokens),
            "test_tokens": len(test_tokens),
            "test_unique_types": len(test_counts),
            "zipf_self_coverage": self_curve,
            "dictionary_hit_rates": scenarios,
        },
        "lexicon_view_do_not_use": lexicon_view,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Natural corpus: {len(dev_tokens):,} dev tokens, "
          f"{len(test_tokens):,} test tokens, "
          f"{len(test_counts):,} unique types\n")

    print("Zipf self-coverage (top-N test types as % of test token volume):")
    print(f"| {'top-N':>8} | {'coverage':>9} |")
    print("|" + "-" * 10 + "|" + "-" * 11 + "|")
    for row in self_curve:
        print(f"| {row['top_n']:>8,} | {row['coverage_pct']:>8.2f}% |")

    print("\nDictionary hit rate against natural test traffic (volume weighted):")
    print(f"| {'dictionary source':<26} | {'entries':>8} | {'volume hit':>11} | {'type hit':>9} |")
    print("|" + "-" * 28 + "|" + "-" * 10 + "|" + "-" * 13 + "|" + "-" * 11 + "|")
    for name, r in scenarios.items():
        print(f"| {name:<26} | {r['dict_entries']:>8,} | "
              f"{r['volume_hit_rate_pct']:>10.2f}% | {r['unique_hit_rate_pct']:>8.2f}% |")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
