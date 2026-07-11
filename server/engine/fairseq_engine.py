"""Stock fairseq XlitEngine, FP32. Quality reference only, not a serving path.

This engine is CPU-only: `ai4bharat.transliteration.XlitEngine` exposes no
device parameter. It is roughly 10x slower than the CTranslate2 engine and its
p95 exceeds the end-to-end latency budget on its own.
"""

from typing import List

from server.compat import stub_urduhack
from server.engine.base import TransliterationEngine, validate_beam

stub_urduhack()

from ai4bharat.transliteration import XlitEngine  # noqa: E402


class FairseqEngine(TransliterationEngine):
    """The unmodified IndicXlit engine, used as the quality baseline."""

    name = "fairseq"

    def __init__(self, lang: str = "hi", beam_width: int = 5, topk: int = 5,
                 rescore: bool = False) -> None:
        validate_beam(beam_width, topk)
        self.lang = lang
        self.beam_width = beam_width
        self.engine = XlitEngine(lang, beam_width=beam_width, rescore=rescore)

    def transliterate(self, word: str, topk: int = 5) -> List[str]:
        validate_beam(self.beam_width, topk)
        return self.engine.translit_word(word, lang_code=self.lang, topk=topk)
