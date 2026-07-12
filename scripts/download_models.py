"""Download the IndicXlit model and stage it into models/ for conversion.

The ai4bharat-transliteration library downloads the fairseq checkpoint on first
use. This script triggers that download and copies the checkpoint, the shared
vocab (corpus-bin), and lang_list.txt into models/indicxlit/fairseq_original/ so
the repo is self-contained and the CT2 conversion can run from a fixed path.

Requires the precompute deps (fairseq + ai4bharat); see
requirements/requirements-precompute.in. Idempotent.

Usage:
    python scripts/download_models.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.compat import stub_urduhack  # noqa: E402

OUT_DIR = "models/indicxlit/fairseq_original"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(f"{OUT_DIR}/indicxlit.pt"):
        print(f"Already present: {OUT_DIR}/indicxlit.pt")
        return

    stub_urduhack()
    from ai4bharat.transliteration import XlitEngine

    print("Constructing XlitEngine('hi') to trigger the model download ...")
    XlitEngine("hi", beam_width=5, rescore=False)

    # The library stores the model under its package dir.
    import ai4bharat.transliteration as pkg
    base = os.path.join(os.path.dirname(pkg.__file__),
                        "transformer/models/en2indic")
    v1 = os.path.join(base, "v1.0")

    shutil.copy2(f"{v1}/transformer/indicxlit.pt", f"{OUT_DIR}/indicxlit.pt")
    if not os.path.isdir(f"{OUT_DIR}/corpus-bin"):
        shutil.copytree(f"{v1}/corpus-bin", f"{OUT_DIR}/corpus-bin")
    # lang_list.txt lives one level up; the CT2 converter needs it in data_dir.
    shutil.copy2(f"{base}/lang_list.txt", f"{OUT_DIR}/corpus-bin/lang_list.txt")

    print(f"Staged checkpoint + vocab + lang_list into {OUT_DIR}/")


if __name__ == "__main__":
    main()
