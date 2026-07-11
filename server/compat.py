"""Compatibility shims for third-party packages with broken dependency chains.

Import this module before importing `ai4bharat.transliteration`.
"""

import sys
import types


def stub_urduhack() -> None:
    """Register a stub `urduhack` module so ai4bharat can be imported.

    `ai4bharat.transliteration.transformer.base_engine` does an unconditional
    `from urduhack import normalize as shahmukhi_normalize` at module load, but
    only calls it when `lang_code == 'ur'`. The real urduhack pulls TensorFlow
    and tf2crf via `urduhack.models.ner`, which will not co-resolve with a
    modern Python 3.10 stack (it pins tensorflow-datasets~=3.1).

    Since this deployment serves Devanagari (Hindi), the symbol is never called.
    The stub raises if it ever is, so an Urdu request fails loudly rather than
    silently skipping normalization.
    """
    if "urduhack" in sys.modules:
        return

    module = types.ModuleType("urduhack")

    def normalize(text: str) -> str:
        raise NotImplementedError(
            "Shahmukhi/Urdu normalization requires the real urduhack package, "
            "which is not installed (it depends on TensorFlow). This deployment "
            "supports Devanagari only."
        )

    module.normalize = normalize
    sys.modules["urduhack"] = module
