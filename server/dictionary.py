"""Dictionary and LRU cache layer.

Serving order is: precomputed dictionary (the head of the distribution), then an
in-process LRU cache of recent model results, then the model itself. This module
owns the first two. It is thread-safe so it can be shared across a worker's
request handlers.
"""

import json
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Source labels returned alongside candidates, matching the API's `source` field.
SOURCE_DICT = "dict"
SOURCE_CACHE = "cache"


class DictionaryCache:
    """Precomputed dictionary plus an LRU cache for model misses.

    Set `dict_path=""` to disable the dictionary and `lru_size=0` to disable the
    cache. With both disabled, every lookup misses and falls through to the model
    (the cache-off ablation).
    """

    def __init__(self, dict_path: str = "", lru_size: int = 10000) -> None:
        self._dict: Dict[str, List[str]] = {}
        self._lru: "OrderedDict[str, List[str]]" = OrderedDict()
        self._lru_size = lru_size
        self._lock = threading.Lock()

        if dict_path:
            with open(dict_path, encoding="utf-8") as handle:
                self._dict = json.load(handle)
            logger.info("Loaded dictionary: %d entries from %s",
                        len(self._dict), dict_path)
        else:
            logger.warning("Dictionary disabled (no dict_path)")

    @property
    def size(self) -> int:
        """Number of precomputed dictionary entries."""
        return len(self._dict)

    def lookup(self, word: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """Return (candidates, source) or (None, None) on a miss.

        Checks the dictionary first, then the LRU cache.
        """
        hit = self._dict.get(word)
        if hit is not None:
            return hit, SOURCE_DICT

        if self._lru_size > 0:
            with self._lock:
                cached = self._lru.get(word)
                if cached is not None:
                    self._lru.move_to_end(word)
                    return cached, SOURCE_CACHE

        return None, None

    def store(self, word: str, candidates: List[str]) -> None:
        """Record a model result in the LRU cache, evicting the oldest if full."""
        if self._lru_size <= 0:
            return
        with self._lock:
            self._lru[word] = candidates
            self._lru.move_to_end(word)
            while len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
