"""Locust load test: Zipfian courtroom-typist simulation.

Each simulated user issues single-word transliteration lookups drawn from a
frequency-weighted vocabulary, so common words dominate exactly as they do in
real typing. The vocabulary is built from Dakshina's natural romanized *test*
corpus, which is held out from the precomputed dictionary (built from train +
dev), so the measured cache-hit ratio is honest.

Per-user request rate is ~0.7 lookups/second (word-boundary debounce at ~40 WPM),
set via `wait_time`. See loadtest/scenarios.md for the scenarios.

Run (from the repo root):
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --users 500 --spawn-rate 50 --run-time 600s --headless \
        --csv loadtest/results/scenario2_int8
"""

import os
import random
import re
from typing import List

from locust import HttpUser, between, task

# Per-user think time (seconds). Default ~0.7 lookups/sec (word-boundary debounce,
# ~40 WPM). Scenario 5 (per-keystroke) overrides these to ~3-5 lookups/sec.
WAIT_MIN = float(os.environ.get("LOADTEST_WAIT_MIN", "1.0"))
WAIT_MAX = float(os.environ.get("LOADTEST_WAIT_MAX", "1.5"))

CORPUS_TEST = os.environ.get(
    "LOADTEST_CORPUS",
    "eval/data/dakshina_dataset_v1.0/hi/romanized/hi.romanized.rejoined.test.roman.txt",
)
TOKEN_RE = re.compile(r"[a-z]+")


def load_weighted_vocab(path: str) -> List[str]:
    """Return a token list with each word repeated by its corpus frequency.

    Sampling uniformly from this list reproduces the natural (Zipfian) word
    distribution without needing an explicit weight array.
    """
    with open(path, encoding="utf-8") as handle:
        tokens = TOKEN_RE.findall(handle.read().lower())
    if not tokens:
        raise RuntimeError(f"no tokens loaded from {path}")
    return tokens


_VOCAB = load_weighted_vocab(CORPUS_TEST)


class Typist(HttpUser):
    """A courtroom typist emitting word-boundary transliteration lookups."""

    # ~0.7 lookups/second per active user (40 WPM with debounce + think time).
    # Override with LOADTEST_WAIT_MIN / LOADTEST_WAIT_MAX for other scenarios.
    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task
    def lookup(self) -> None:
        word = random.choice(_VOCAB)
        # name= groups all lookups under one entry in Locust stats.
        self.client.get(
            f"/transliterate?word={word}&lang=hi&topk=5",
            name="/transliterate",
        )
