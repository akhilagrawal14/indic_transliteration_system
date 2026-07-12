"""Download and extract the Dakshina Hindi data used for eval and the dictionary.

Fetches the Dakshina v1.0 archive and extracts just the Hindi lexicons and the
natural romanized corpus (the rest of the ~2 GB archive is not needed). Idempotent:
skips downloads/extractions that already exist.

Usage:
    python scripts/download_data.py
"""

import os
import subprocess
import sys
import tarfile

DATA_DIR = "eval/data"
ARCHIVE = f"{DATA_DIR}/dakshina_dataset_v1.0.tar"
URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
WANTED_PREFIXES = (
    "dakshina_dataset_v1.0/hi/lexicons/",
    "dakshina_dataset_v1.0/hi/romanized/",
)
SENTINEL = f"{DATA_DIR}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SENTINEL):
        print(f"Already extracted: {SENTINEL}")
        return

    if not os.path.exists(ARCHIVE):
        print(f"Downloading Dakshina (~2 GB) to {ARCHIVE} ...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", ARCHIVE, URL],
                       check=True)

    print("Extracting Hindi lexicons + romanized corpus ...")
    with tarfile.open(ARCHIVE) as tar:
        members = [m for m in tar.getmembers()
                   if m.name.startswith(WANTED_PREFIXES)]
        if not members:
            sys.exit("No Hindi members found in the archive")
        tar.extractall(DATA_DIR, members=members)

    print(f"Done. Lexicons + romanized corpus under {DATA_DIR}/dakshina_dataset_v1.0/hi/")


if __name__ == "__main__":
    main()
