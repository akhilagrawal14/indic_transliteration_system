"""Convert the IndicXlit fairseq checkpoint to CTranslate2 INT8.

Wraps ct2-fairseq-converter with the three flags this multilingual checkpoint
needs (learned the hard way, see docs/architecture.md ADR / the report):
  --unsafe_deserialization : the checkpoint pickles an argparse.Namespace
  lang_list.txt in the data dir : required by translation_multi_simple_epoch
  --source_lang en --target_lang hi : selects the shared vocab

Run scripts/download_models.py first. Idempotent.

Usage:
    python scripts/convert_ct2.py
"""

import argparse
import os
import shutil
import subprocess

FAIRSEQ_DIR = "models/indicxlit/fairseq_original"
DATA_DIR = f"{FAIRSEQ_DIR}/corpus-bin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/indicxlit/ct2_int8")
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="hi")
    args = parser.parse_args()

    if os.path.exists(f"{args.output_dir}/model.bin"):
        print(f"Already converted: {args.output_dir}/model.bin")
        return
    if not os.path.exists(f"{FAIRSEQ_DIR}/indicxlit.pt"):
        raise SystemExit("Run scripts/download_models.py first")

    # lang_list.txt must sit inside the data dir.
    if not os.path.exists(f"{DATA_DIR}/lang_list.txt"):
        up = f"{FAIRSEQ_DIR}/lang_list.txt"
        if os.path.exists(up):
            shutil.copy2(up, f"{DATA_DIR}/lang_list.txt")

    cmd = [
        "ct2-fairseq-converter",
        "--model_path", f"{FAIRSEQ_DIR}/indicxlit.pt",
        "--data_dir", DATA_DIR,
        "--source_lang", args.source_lang,
        "--target_lang", args.target_lang,
        "--quantization", args.quantization,
        "--unsafe_deserialization",
        "--force",
        "--output_dir", args.output_dir,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Converted to {args.output_dir}/ ({args.quantization})")


if __name__ == "__main__":
    main()
