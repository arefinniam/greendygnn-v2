#!/usr/bin/env bash
###############################################################################
# prepare_optisched.sh — build the offline OptiSched artifacts (Phase 0/2).
#
# For each dataset x batch size it:
#   1. replays the seeded sampler to DUMP the deterministic trace (dump_trace.py,
#      launched distributed exactly like the trainers), then
#   2. builds the online regime LIBRARY and the clairvoyant DP SCHEDULE
#      (build_library.py) into $OPTISCHED_LIB_DIR, which run_benchmark.sh reads
#      for the `optisched` / `optisched_dp` methods.
#
# Usage:
#   DATASET_ROOT=/path ./prepare_optisched.sh [--datasets ...] [--batch-sizes ...]
#                                              [--trace-epochs 5] [--n-hot 100000]
# Optional:
#   OPTISCHED_MODEL=calib.json   calibrated CostModel (else built-in defaults)
#   OPTISCHED_LIB_DIR=./libraries
###############################################################################
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_CONFIG="$DIR/ip_config.txt"
: "${DATASET_ROOT:?DATASET_ROOT must be set}"
OPTISCHED_LIB_DIR="${OPTISCHED_LIB_DIR:-$DIR/libraries}"
TRACE_DIR="$DIR/traces"
DATASETS="ogbn-products,reddit,ogbn-papers100M"
BATCHES="1000,2000,3000"
TRACE_EPOCHS=5
N_HOT=100000
W_MAX=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)     DATASETS="$2"; shift 2 ;;
        --batch-sizes)  BATCHES="$2"; shift 2 ;;
        --trace-epochs) TRACE_EPOCHS="$2"; shift 2 ;;
        --n-hot)        N_HOT="$2"; shift 2 ;;
        --w-max)        W_MAX="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

declare -A PART_CONFIGS
PART_CONFIGS["ogbn-products"]="$DATASET_ROOT/OGBN-Products/data/ogbn-products.json"
PART_CONFIGS["reddit"]="$DATASET_ROOT/Reddit/data/reddit.json"
PART_CONFIGS["ogbn-papers100M"]="$DATASET_ROOT/OGBN-Papers/data/ogbn-papers100M.json"

MODEL_ARG=""
[[ -n "${OPTISCHED_MODEL:-}" ]] && MODEL_ARG="--model $OPTISCHED_MODEL"

mkdir -p "$OPTISCHED_LIB_DIR" "$TRACE_DIR"
IFS=',' read -ra D_ARR <<< "$DATASETS"
IFS=',' read -ra B_ARR <<< "$BATCHES"

for dataset in "${D_ARR[@]}"; do
    pc="${PART_CONFIGS[$dataset]}"
    for batch in "${B_ARR[@]}"; do
        echo "── prepare $dataset B$batch ──"
        # 1. dump the trace (distributed, via launch.py)
        python3 "$DIR/launch.py" --workspace "$DIR" \
            --num_trainers 1 --num_servers 1 \
            --part_config "$pc" --ip_config "$IP_CONFIG" \
            "python3 $DIR/dump_trace.py --graph_name $dataset --ip_config $IP_CONFIG \
                --part_config $pc --batch_size $batch --num_epochs $TRACE_EPOCHS \
                --out $TRACE_DIR --seed 1" \
            || { echo "  dump failed for $dataset B$batch"; continue; }

        # 2. build online library + clairvoyant schedule (local, single process)
        python3 "$DIR/build_library.py" --traces "$TRACE_DIR" --dataset "$dataset" \
            --part 0 $MODEL_ARG --n_hot "$N_HOT" --w_max "$W_MAX" --mode library \
            --out "$OPTISCHED_LIB_DIR/${dataset}_B${batch}_lib.json"
        python3 "$DIR/build_library.py" --traces "$TRACE_DIR" --dataset "$dataset" \
            --part 0 $MODEL_ARG --n_hot "$N_HOT" --w_max "$W_MAX" --mode clairvoyant \
            --out "$OPTISCHED_LIB_DIR/${dataset}_B${batch}_dp.json"
    done
done
echo "OptiSched artifacts ready in $OPTISCHED_LIB_DIR"
