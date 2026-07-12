"""Engine tests. The CT2 test is an integration test: it skips if the converted
model is not present (a fresh clone before the conversion step)."""

import os

import pytest

from server.engine.base import validate_beam

CT2_DIR = "models/indicxlit/ct2_int8"


def test_validate_beam_raises_when_below_topk():
    with pytest.raises(ValueError):
        validate_beam(beam_width=4, topk=5)
    validate_beam(beam_width=5, topk=5)  # ok, no raise


@pytest.mark.skipif(not os.path.isdir(CT2_DIR),
                    reason="CT2 model not converted yet")
def test_ct2_engine_returns_ranked_candidates():
    from server.engine.ct2_engine import CT2Engine
    engine = CT2Engine(CT2_DIR, lang="hi", beam_width=5, topk=5, device="cpu")
    cands = engine.transliterate("mera", topk=5)
    assert len(cands) == 5
    assert cands[0] == "मेरा"           # known top-1 for this model
    assert all(isinstance(c, str) and c for c in cands)


@pytest.mark.skipif(not os.path.isdir(CT2_DIR),
                    reason="CT2 model not converted yet")
def test_ct2_engine_rejects_topk_above_beam():
    from server.engine.ct2_engine import CT2Engine
    with pytest.raises(ValueError):
        CT2Engine(CT2_DIR, lang="hi", beam_width=3, topk=5, device="cpu")
