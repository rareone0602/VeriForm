#!/usr/bin/env bash
# CCDS-side incremental upload of new pickles to GCS, for the migration
# bridge. Designed to run from sbatch (NOT the head node — pickles can be
# large enough that gcloud's hashing adds up).
#
# Usage:
#   bash scripts/sync_to_gcs.sh         # one-shot
#   bash scripts/sync_to_gcs.sh --loop  # rsync every 30 min
#
# Expects gcloud SDK at /export/home3/phy/google-cloud-sdk/ (CCDS).

set -euo pipefail

export PATH="/export/home3/phy/google-cloud-sdk/bin:$PATH"
BUCKET="gs://veriform-faithformbench-2026"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Sanity
command -v gcloud >/dev/null || { echo "gcloud missing" >&2; exit 1; }
[ "$(gcloud config get-value project 2>/dev/null)" = "research-496610" ] || \
    { echo "wrong gcloud project; set research-496610" >&2; exit 1; }

LOOP=0
if [ "${1:-}" = "--loop" ]; then LOOP=1; fi

do_sync() {
    echo "=== sync at $(date -Is) ==="
    # Pickles (formalized + proved) for all methods. Skip overleaf/, venv/,
    # __pycache__/, etc — only data/.
    gcloud storage rsync -r --gzip-in-flight-all \
        data/output/ "$BUCKET/data/output/" 2>&1 | tail -8
    # DAG inputs (small, rarely change)
    gcloud storage rsync -r \
        data/parsed/ "$BUCKET/data/parsed/" 2>&1 | tail -3
    # Paper baselines (.dat heatmap files)
    if [ -d overleaf/latex/assets/FaithformBench ]; then
        gcloud storage rsync -r \
            overleaf/latex/assets/FaithformBench/ \
            "$BUCKET/paper/FaithformBench/" 2>&1 | tail -3
    fi
    echo "=== done at $(date -Is) ==="
}

if [ "$LOOP" = "1" ]; then
    while true; do
        do_sync || echo "sync error (continuing)"
        sleep 1800  # 30 min
    done
else
    do_sync
fi
