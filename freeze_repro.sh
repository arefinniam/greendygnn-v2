#!/usr/bin/env bash
###############################################################################
# freeze_repro.sh — freeze reproducibility metadata BEFORE the cluster gate run.
#
# Binds the real-trace result to the exact rules + code it was read under:
# content hashes of the pre-registration and ledger (the "git hash" substitute
# in this non-git copy), a combined code hash, the calibration config, the fixed
# experimental knobs, machine/node identity, and a timestamp.
#
# Run it now (offline) to capture hashes+config, and again on the cluster right
# before launching so it also records node identity + the calibration hash.
#
# Usage:
#   ./freeze_repro.sh [--model calib.json] [--datasets reddit,ogbn-products,...] \
#                     [--n-hot 100000] [--stretch-lens 2,3,5,7,10,15] [--out FILE]
###############################################################################
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=""
DATASETS="reddit,ogbn-products,ogbn-papers100M"
N_HOT=100000
STRETCH_LENS="2,3,5,7,10,15"
W_MAX=64
OUT="$DIR/results/RUN_MANIFEST.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --datasets)     DATASETS="$2"; shift 2 ;;
        --n-hot)        N_HOT="$2"; shift 2 ;;
        --stretch-lens) STRETCH_LENS="$2"; shift 2 ;;
        --w-max)        W_MAX="$2"; shift 2 ;;
        --out)          OUT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$(dirname "$OUT")"

h() { [[ -f "$1" ]] && sha256sum "$1" | cut -d' ' -f1 || echo "MISSING"; }

# Combined code hash: all algorithm sources, order-stable.
CODE_FILES=$(ls "$DIR"/optisched/*.py "$DIR"/run_gate.py "$DIR"/dump_trace.py \
    "$DIR"/build_library.py "$DIR"/calibrate.py "$DIR"/train_optisched.py \
    "$DIR"/prefetcher.py "$DIR"/tests/test_optisched.py 2>/dev/null | sort)
CODE_HASH=$(cat $CODE_FILES 2>/dev/null | sha256sum | cut -d' ' -f1)

GIT_HASH=$(git -C "$DIR" rev-parse HEAD 2>/dev/null || echo "none-not-a-git-repo")
PREREG_HASH=$(h "$DIR/GATE_PREREGISTRATION.md")
LEDGER_HASH=$(h "$DIR/OPTISCHED_LEDGER.md")
MODEL_HASH=$([[ -n "$MODEL" ]] && h "$MODEL" || echo "not-yet-calibrated")
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# JSON array of per-file hashes for auditability.
FILE_HASHES=$(for f in $CODE_FILES; do
    printf '    {"file": "%s", "sha256": "%s"},\n' "$(basename "$f")" "$(h "$f")"
done | sed '$ s/,$//')

cat > "$OUT" <<JSON
{
  "frozen_at_utc": "$TS",
  "purpose": "reproducibility freeze for the OptiSched-GNN real-trace cluster gate",
  "git_hash": "$GIT_HASH",
  "preregistration_sha256": "$PREREG_HASH",
  "ledger_sha256": "$LEDGER_HASH",
  "code_combined_sha256": "$CODE_HASH",
  "calibration_model": "${MODEL:-not-yet-calibrated}",
  "calibration_sha256": "$MODEL_HASH",
  "config": {
    "n_hot": $N_HOT,
    "stretch_lens": [$(echo "$STRETCH_LENS" | sed 's/,/, /g')],
    "w_max": $W_MAX,
    "datasets": ["$(echo "$DATASETS" | sed 's/,/", "/g')"]
  },
  "machine": {
    "hostname": "$(hostname 2>/dev/null || echo unknown)",
    "uname": "$(uname -srm 2>/dev/null || echo unknown)",
    "node_ids_note": "fill at cluster launch: the 4 gnn node IDs from ip_config.txt"
  },
  "code_files": [
$FILE_HASHES
  ]
}
JSON

echo "Reproducibility manifest -> $OUT"
echo "  prereg  sha256: $PREREG_HASH"
echo "  ledger  sha256: $LEDGER_HASH"
echo "  code    sha256: $CODE_HASH"
echo "  git           : $GIT_HASH"
echo "  calibration   : ${MODEL:-not-yet-calibrated} ($MODEL_HASH)"
echo "  config        : n_hot=$N_HOT stretch_lens=$STRETCH_LENS w_max=$W_MAX"
echo ""
echo "Re-run on the cluster (with --model calib.json) right before launching so it"
echo "also records the calibration hash and node identity."
