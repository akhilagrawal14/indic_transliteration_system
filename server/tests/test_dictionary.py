"""Tests for the dictionary + LRU cache layer."""

import json

from server.dictionary import SOURCE_CACHE, SOURCE_DICT, DictionaryCache


def _write_dict(tmp_path, mapping):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


def test_dictionary_hit(tmp_path):
    cache = DictionaryCache(_write_dict(tmp_path, {"mera": ["मेरा", "मीरा"]}))
    cands, source = cache.lookup("mera")
    assert cands == ["मेरा", "मीरा"]
    assert source == SOURCE_DICT


def test_miss_returns_none(tmp_path):
    cache = DictionaryCache(_write_dict(tmp_path, {"mera": ["मेरा"]}))
    assert cache.lookup("unknownword") == (None, None)


def test_lru_store_then_hit(tmp_path):
    cache = DictionaryCache(_write_dict(tmp_path, {}), lru_size=10)
    assert cache.lookup("zzz") == (None, None)
    cache.store("zzz", ["ज़ज़"])
    cands, source = cache.lookup("zzz")
    assert cands == ["ज़ज़"]
    assert source == SOURCE_CACHE


def test_dictionary_precedes_cache(tmp_path):
    cache = DictionaryCache(_write_dict(tmp_path, {"a": ["dict"]}), lru_size=10)
    cache.store("a", ["cache"])  # should never win over the dictionary
    cands, source = cache.lookup("a")
    assert source == SOURCE_DICT
    assert cands == ["dict"]


def test_lru_eviction(tmp_path):
    cache = DictionaryCache(_write_dict(tmp_path, {}), lru_size=2)
    cache.store("a", ["1"])
    cache.store("b", ["2"])
    cache.store("c", ["3"])  # evicts "a" (oldest)
    assert cache.lookup("a") == (None, None)
    assert cache.lookup("c")[0] == ["3"]


def test_disabled_dictionary_and_cache():
    cache = DictionaryCache(dict_path="", lru_size=0)
    assert cache.size == 0
    cache.store("a", ["x"])          # no-op when lru disabled
    assert cache.lookup("a") == (None, None)
