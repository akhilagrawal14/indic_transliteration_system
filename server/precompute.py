"""Precompute the transliteration dictionary on CPU.

Runs the CTranslate2 INT8 engine over the union of the Dakshina train-lexicon
romanizations and the natural-corpus dev vocabulary (the ~52k input set that
measured an 89.5% held-out hit rate), and writes `roman -> [ranked candidates]`
to a JSON file the server loads at startup.

CPU only. No GPU is needed: CTranslate2 uses all cores for the batch, so ~50k
short words convert in seconds. Run once per language/model.

Usage:
    python server/precompute.py
    python server/precompute.py --output server/data/dictionary_hi.json --topk 5
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.engine.ct2_engine import CT2Engine  # noqa: E402

BASE = "eval/data/dakshina_dataset_v1.0/hi"
LEXICON_TRAIN = f"{BASE}/lexicons/hi.translit.sampled.train.tsv"
CORPUS_DEV = f"{BASE}/romanized/hi.romanized.rejoined.dev.roman.txt"
TOKEN_RE = re.compile(r"[a-z]+")


def lexicon_romanizations(path: str) -> Set[str]:
    """Unique romanized inputs from a Dakshina lexicon TSV (native, roman, count)."""
    inputs: Set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                inputs.add(parts[1].lower())
    return inputs


def corpus_vocab(path: str) -> Set[str]:
    """Unique lowercase word tokens from a natural romanized text file."""
    with open(path, encoding="utf-8") as handle:
        return set(TOKEN_RE.findall(handle.read().lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/indicxlit/ct2_int8")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default="server/data/dictionary_hi.json")
    args = parser.parse_args()

    inputs = sorted(lexicon_romanizations(LEXICON_TRAIN) | corpus_vocab(CORPUS_DEV))
    print(f"Building dictionary for {len(inputs):,} unique inputs "
          f"(beam={args.beam}, topk={args.topk}) on CPU...")

    # intra_threads=0 lets CTranslate2 use all cores for the batch job.
    engine = CT2Engine(args.model_dir, lang=args.lang, beam_width=args.beam,
                       topk=args.topk, device="cpu", intra_threads=0)

    dictionary = {}
    start = time.perf_counter()
    for i in range(0, len(inputs), args.batch_size):
        batch = inputs[i: i + args.batch_size]
        for word, cands in zip(batch, engine.transliterate_batch(batch, args.topk)):
            dictionary[word] = cands
        if i % (args.batch_size * 10) == 0:
            print(f"  {i:,}/{len(inputs):,}", flush=True)
    elapsed = time.perf_counter() - start

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(dictionary, handle, ensure_ascii=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {len(dictionary):,} entries to {args.output} "
          f"({size_mb:.1f} MB) in {elapsed:.1f}s "
          f"({elapsed / len(inputs) * 1000:.2f} ms/word)")


if __name__ == "__main__":
    main()
