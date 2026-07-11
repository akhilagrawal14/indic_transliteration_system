#!/usr/bin/env bash
# One-command start for the Indic Transliteration Runtime.
#
# Usage:
#   ./run.sh docker    # build and run backend + frontend via docker compose
#   ./run.sh dev       # run backend (uvicorn) and frontend (npm) locally
#
# Local dev assumes the conda env `xlit` exists and the demo has been
# initialized (see docs/setup.md).

set -euo pipefail

MODE="${1:-docker}"

case "$MODE" in
  docker)
    echo "Starting backend + frontend via docker compose..."
    docker compose up --build
    ;;

  dev)
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xlit

    echo "Starting backend on :${PORT:-8000}..."
    uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${WORKERS:-4}" --reload &
    BACKEND_PID=$!

    echo "Starting frontend on :3000..."
    (cd demo && npm run dev) &
    FRONTEND_PID=$!

    # Stop both children on Ctrl-C
    trap 'kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true' INT TERM
    wait
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: ./run.sh [docker|dev]" >&2
    exit 1
    ;;
esac
