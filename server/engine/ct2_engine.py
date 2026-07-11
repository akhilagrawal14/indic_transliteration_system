"""CTranslate2 INT8 engine. This is the production serving path."""

from typing import List

import ctranslate2

from server.engine.base import TransliterationEngine, validate_beam


class CT2Engine(TransliterationEngine):
    """IndicXlit converted to CTranslate2, quantized to INT8.

    The multilingual model expects a target-language tag prepended to the
    character sequence, e.g. `mera` becomes `["__hi__", "m", "e", "r", "a"]`.
    CT2 appends `</s>` itself via `add_source_eos` in the model config.
    """

    name = "ct2"

    def __init__(self, model_dir: str, lang: str = "hi", beam_width: int = 5,
                 topk: int = 5, device: str = "cpu", intra_threads: int = 1,
                 compute_type: str = "int8") -> None:
        validate_beam(beam_width, topk)
        self.lang = lang
        self.beam_width = beam_width
        self.translator = ctranslate2.Translator(
            model_dir, device=device, compute_type=compute_type,
            intra_threads=intra_threads,
        )

    def _encode(self, word: str) -> List[str]:
        return [f"__{self.lang}__"] + list(word.lower())

    def transliterate(self, word: str, topk: int = 5) -> List[str]:
        return self.transliterate_batch([word], topk)[0]

    def transliterate_batch(self, words: List[str], topk: int = 5) -> List[List[str]]:
        validate_beam(self.beam_width, topk)
        results = self.translator.translate_batch(
            [self._encode(w) for w in words],
            beam_size=self.beam_width,
            num_hypotheses=topk,
        )
        return [["".join(h) for h in r.hypotheses] for r in results]
