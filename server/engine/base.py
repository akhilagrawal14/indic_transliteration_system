"""Abstract interface for transliteration engines.

Adding a new engine (ONNX, TensorRT, plain PyTorch) means implementing this one
class. The serving layer and the eval harness depend only on this interface.
"""

from abc import ABC, abstractmethod
from typing import List


class TransliterationEngine(ABC):
    """Maps a romanized word to a ranked list of Indic-script candidates."""

    name: str

    @abstractmethod
    def transliterate(self, word: str, topk: int = 5) -> List[str]:
        """Return up to `topk` candidates for `word`, best first."""

    def transliterate_batch(self, words: List[str], topk: int = 5) -> List[List[str]]:
        """Transliterate several words. Engines with native batching override this."""
        return [self.transliterate(word, topk) for word in words]


def validate_beam(beam_width: int, topk: int) -> None:
    """Fail loudly when beam width would silently truncate the candidate list.

    Beam width caps the number of hypotheses the decoder can return, so a
    `beam_width` below `topk` yields fewer candidates than requested.
    """
    if beam_width < topk:
        raise ValueError(
            f"beam_width ({beam_width}) must be >= topk ({topk}), otherwise the "
            f"engine silently returns only {beam_width} candidates."
        )
