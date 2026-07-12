#!/usr/bin/env bash
# One-time artifact preparation for a fresh clone: data, model, CT2 conversion,
# and the precomputed dictionary. Run inside the `xlit` conda env with the
# precompute deps installed (see requirements/requirements-precompute.in).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/download_data.py       # Dakshina Hindi lexicons + corpus
python scripts/download_models.py     # IndicXlit fairseq checkpoint -> models/
python scripts/convert_ct2.py         # fairseq -> CTranslate2 INT8
python server/precompute.py           # build server/data/dictionary_hi.json
python scripts/build_client_dict.py   # demo/public/client_dict_hi.json (offline path)

echo "Bootstrap complete. Now: docker compose up --build  (or ./run.sh dev)"
