#!/usr/bin/env bash
###############################################################################
# run_matrix.sh — GreenDyGNN v2 full campaign driver (runs ON gnn1).
#
# {dataset} x {method} x {condition} x {seed}, sequential DistDGL jobs.
# Per run: cluster cleanup barrier -> RAPL chmod barrier -> tc-clean verify ->
# condition apply (steady) or co-process (squarewave / organic) -> launch via
# launch.py (UDF positional BEFORE --extra_envs) -> collect 4 part profiles
# with retry -> aggregate (aggregate_p1.py, wrap-robust median-of-positive-
# deltas) -> append matrix row. Resumable: a run whose p1_aggregate.json
# already exists is skipped. Seed order within each (condition,method) block
# is shuffled deterministically and logged (plan.txt) to decorrelate drift.
#
# Usage:
#   bash run_matrix.sh [--dry-run] [--datasets reddit,ogbn-products]
#        [--methods default_dgl,rapidgnn_epoch,static_w16,greendygnn_v2,v2_no_rl,v2_uniform_alloc]
#        [--conditions clean,c1_200,c1_100,c1_50,c2_duty,c3_sq200]
#        [--seeds 0,1,2] [--epochs 30] [--batch 2000] [--warmup 5]
#        [--victim gnn4] [--out DIR] [--timeout 3600] [--verify]
#
# Conditions:
#   clean      no impairment (tc verified clean)
#   c1_<R>     steady tbf <R>mbit on victim egress (broad eno1 root)
#   c2_duty    organic iperf3 cross-traffic, duty-cycled 30s/30s, 8 streams, incast
#   c3_sq<R>   square-wave tbf <R>mbit, period 120s, duty 0.5 (wall-clock)
###############################################################################
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGG="${AGG_PATH:-$HOME/optisched_sys/aggregate_p1.py}"
GDY_CKPT="${GDY_CKPT:-$DIR/checkpoints/greendygnn_dqn.pt}"
CACHE_SIZE="${CACHE_SIZE:-100000}"

DATASETS="reddit,ogbn-products"
METHODS="default_dgl,rapidgnn_epoch,static_w16,greendygnn_v2,v2_no_rl,v2_uniform_alloc"
CONDITIONS="clean,c1_200,c1_100,c1_50,c2_duty,c3_sq200"
SEEDS="0,1,2"
EPOCHS=30
BATCH=2000
WARMUP=5
VICTIM="gnn4"
TIMEOUT_PER_RUN=3600
OUT_ROOT=""
DRY_RUN=0
DO_VERIFY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --datasets)   DATASETS="$2"; shift 2 ;;
        --methods)    METHODS="$2"; shift 2 ;;
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --victim)     VICTIM="$2"; shift 2 ;;
        --out)        OUT_ROOT="$2"; shift 2 ;;
        --timeout)    TIMEOUT_PER_RUN="$2"; shift 2 ;;
        --verify)     DO_VERIFY=1; shift ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

OUT_ROOT="${OUT_ROOT:-$HOME/matrix_results/matrix_$(date +%Y%m%d_%H%M%S)}"
RUNS_DIR="$OUT_ROOT/runs"
MATRIX_TSV="$OUT_ROOT/matrix.tsv"
PLAN="$OUT_ROOT/plan.txt"

# ── Dataset metadata ─────────────────────────────────────────────────────────
declare -A PART_CONFIGS NCLASSES
PART_CONFIGS["reddit"]="$HOME/distgnn/data/reddit/reddit.json"
PART_CONFIGS["ogbn-products"]="$HOME/distgnn/data/ogbn-products/ogbn-products.json"
NCLASSES["reddit"]=41
NCLASSES["ogbn-products"]=47

# Node registry (private = eno1)
declare -A PRIV
PRIV["gnn1"]="10.52.2.119"; PRIV["gnn2"]="10.52.3.217"
PRIV["gnn3"]="10.52.3.123"; PRIV["gnn4"]="10.52.3.89"
ALL_NODES="gnn1 gnn2 gnn3 gnn4"
PEER_SSH="ssh -i $HOME/.ssh/peerkey.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10"

TRAIN_COMMON="--num_gpus 1 --num_hidden 16 --num_layers 2 --fan_out 10,25 \
--lr 0.003 --dropout 0.5 --log_every 20"

# ── helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

on_node() {  # on_node <node> <cmd>
    local n="$1"; shift
    if [[ "$(hostname -s)" == "$n" ]]; then bash -c "$*"
    else $PEER_SSH "cc@${PRIV[$n]}" "$*"; fi
}

# Deterministic per-block seed shuffle (logged): shuffle via sha-sum ordering.
shuffle_seeds() {  # shuffle_seeds <key> <s1,s2,...>
    local key="$1"; local seeds="$2"
    echo "$seeds" | tr ',' '\n' | while read -r s; do
        echo "$(printf '%s|%s' "$key" "$s" | sha1sum | cut -c1-8) $s"
    done | sort | awk '{print $2}' | paste -sd' ' -
}

cleanup_barrier() {
    log "cleanup barrier (procs + ports + tc)"
    for n in $ALL_NODES; do
        on_node "$n" '
            pkill -9 python3 2>/dev/null || true
            for p in 30050 30051 30052 30053 29500 29501 1234; do
                fuser -k ${p}/tcp 2>/dev/null || true
            done' || true
    done
    sleep 2
    for n in $ALL_NODES; do
        local c
        c=$(on_node "$n" 'pgrep -c -f "train_|launch.py" 2>/dev/null || echo 0' | tail -1)
        if [[ "${c:-0}" != "0" ]]; then
            log "WARN: $n still has $c training procs after cleanup; retrying kill"
            on_node "$n" 'pkill -9 -f "train_|launch.py" 2>/dev/null || true'
        fi
    done
    python3 "$DIR/congestion2.py" teardown --ssh-mode peer || {
        log "FATAL: tc teardown failed"; exit 1; }
}

rapl_barrier() {
    log "RAPL readable barrier"
    for n in $ALL_NODES; do
        local ok=0
        for _ in 1 2 3 4 5; do
            if on_node "$n" '
                sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj \
                    /sys/class/powercap/intel-rapl*/max_energy_range_uj \
                    /sys/class/powercap/intel-rapl*/name 2>/dev/null
                head -c1 /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1'; then
                ok=1; break
            fi
            sleep 2
        done
        [[ $ok -eq 1 ]] || { log "FATAL: RAPL unreadable on $n"; exit 1; }
    done
}

build_train_cmd() {  # build_train_cmd <method> <dataset> <seed> <run_dir>
    local method="$1" dataset="$2" seed="$3" rd="$4"
    local pc="${PART_CONFIGS[$dataset]}" nc="${NCLASSES[$dataset]}"
    local common="--graph_name $dataset --ip_config $IP_CONFIG --part_config $pc \
--num_epochs $EPOCHS --batch_size $BATCH --out_dir $rd --seed $seed \
--n_classes $nc $TRAIN_COMMON"
    local ckpt_flag=""
    [[ -f "$GDY_CKPT" ]] && ckpt_flag="--checkpoint $GDY_CKPT"
    case "$method" in
        default_dgl)      echo "python3 $DIR/train_default.py $common" ;;
        rapidgnn_epoch)   echo "python3 $DIR/train_rapidgnn.py $common --window_size 0 --cache_size $CACHE_SIZE" ;;
        static_w*)        local wn="${method#static_w}"
                          [[ "$wn" =~ ^[0-9]+$ ]] || { echo "ERROR: bad static_w method '$method'" >&2; return 1; }
                          echo "python3 $DIR/train_rapidgnn.py $common --window_size $wn --cache_size $CACHE_SIZE" ;;
        greendygnn_v2)    echo "python3 $DIR/train_greendygnn.py $common --window_size 16 --cache_size $CACHE_SIZE $ckpt_flag" ;;
        v2_no_rl)         echo "python3 $DIR/train_greendygnn.py $common --window_size 16 --cache_size $CACHE_SIZE $ckpt_flag --no_rl" ;;
        v2_uniform_alloc) echo "python3 $DIR/train_greendygnn.py $common --window_size 16 --cache_size $CACHE_SIZE $ckpt_flag --uniform_alloc" ;;
        *) echo "ERROR: unknown method $method" >&2; return 1 ;;
    esac
}

# Condition handling. Sets globals COND_STEADY_APPLIED and COND_BG_PID.
condition_start() {  # condition_start <condition> <run_dir>
    local cond="$1" rd="$2"
    COND_STEADY_APPLIED=0; COND_BG_PID=""
    case "$cond" in
        clean)
            python3 "$DIR/congestion2.py" verify --ssh-mode peer \
                --journal "$rd/cong_journal.jsonl" || exit 1 ;;
        c1_*)
            local rate="${cond#c1_}"
            python3 "$DIR/congestion2.py" apply --cls c1 --victims "$VICTIM" \
                --rate "${rate}mbit" --ssh-mode peer \
                --journal "$rd/cong_journal.jsonl" || exit 1
            COND_STEADY_APPLIED=1 ;;
        c3_sq*)
            local rate="${cond#c3_sq}"
            python3 "$DIR/congestion2.py" run --cls c1 --mode squarewave \
                --rate "${rate}mbit" --victims "$VICTIM" \
                --period 120 --duty 0.5 --duration "$TIMEOUT_PER_RUN" \
                --ssh-mode peer --journal "$rd/cong_journal.jsonl" \
                > "$rd/congestion.log" 2>&1 &
            COND_BG_PID=$! ;;
        c2_duty)
            python3 "$DIR/congestion2.py" run --cls c2 --victims "$VICTIM" \
                --on 30 --off 30 --streams 8 --incast \
                --duration "$TIMEOUT_PER_RUN" --ssh-mode peer \
                --journal "$rd/cong_journal.jsonl" \
                > "$rd/congestion.log" 2>&1 &
            COND_BG_PID=$! ;;
        *) log "FATAL: unknown condition $cond"; exit 1 ;;
    esac
}

condition_stop() {  # condition_stop <run_dir>
    local rd="$1"
    if [[ -n "${COND_BG_PID}" ]]; then
        kill -TERM "$COND_BG_PID" 2>/dev/null || true
        wait "$COND_BG_PID" 2>/dev/null || true
    fi
    python3 "$DIR/congestion2.py" teardown --ssh-mode peer \
        --journal "$rd/cong_journal.jsonl" || {
        log "FATAL: post-run teardown failed"; exit 1; }
}

collect_profiles() {  # collect_profiles <run_dir> -> 0 iff 4 parts present
    local rd="$1"
    for attempt in 1 2 3 4 5 6; do
        for n in gnn2 gnn3 gnn4; do
            scp -i "$HOME/.ssh/peerkey.pem" -o StrictHostKeyChecking=no -q \
                "cc@${PRIV[$n]}:$rd/*_part*_profile.json" "$rd/" 2>/dev/null || true
        done
        local c
        c=$(ls "$rd"/*_part*_profile.json 2>/dev/null | wc -l)
        [[ "$c" -ge 4 ]] && return 0
        log "  collect attempt $attempt: $c/4 profiles; retrying in 10s"
        sleep 10
    done
    return 1
}

# ── Build the plan ───────────────────────────────────────────────────────────
IFS=',' read -ra D_ARR <<< "$DATASETS"
IFS=',' read -ra M_ARR <<< "$METHODS"
IFS=',' read -ra C_ARR <<< "$CONDITIONS"

mkdir -p "$RUNS_DIR"
: > "$PLAN"
declare -a PLAN_ROWS=()
for dataset in "${D_ARR[@]}"; do
    for cond in "${C_ARR[@]}"; do
        for method in "${M_ARR[@]}"; do
            for seed in $(shuffle_seeds "$dataset|$cond|$method" "$SEEDS"); do
                PLAN_ROWS+=("$dataset|$cond|$method|$seed")
                echo "$dataset|$cond|$method|seed$seed" >> "$PLAN"
            done
        done
    done
done
TOTAL=${#PLAN_ROWS[@]}

echo "════════════════════════════════════════════════════════════════════"
echo " GreenDyGNN v2 matrix: $TOTAL runs"
echo "  datasets   : $DATASETS"
echo "  methods    : $METHODS"
echo "  conditions : $CONDITIONS  (victim: $VICTIM)"
echo "  seeds      : $SEEDS (order shuffled per block, see $PLAN)"
echo "  epochs=$EPOCHS batch=$BATCH warmup=$WARMUP timeout=${TIMEOUT_PER_RUN}s"
echo "  out        : $OUT_ROOT"
echo "  checkpoint : $GDY_CKPT $( [[ -f $GDY_CKPT ]] && echo '(found)' || echo '(MISSING -> controller falls back to heuristic)')"
echo "════════════════════════════════════════════════════════════════════"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "-- DRY RUN: planned runs and commands --"
    IP_CONFIG="$OUT_ROOT/ip_config.txt"   # placeholder path in printout
    i=0
    for row in "${PLAN_ROWS[@]}"; do
        IFS='|' read -r dataset cond method seed <<< "$row"
        i=$((i+1))
        rd="$RUNS_DIR/${dataset}__${cond}__${method}__seed${seed}"
        echo ""
        echo "[$i/$TOTAL] $dataset / $cond / $method / seed$seed"
        echo "  run_dir: $rd"
        echo "  cmd: $(build_train_cmd "$method" "$dataset" "$seed" "$rd")"
        case "$cond" in
            clean)  echo "  congestion: none (verified clean)" ;;
            c1_*)   echo "  congestion: steady tbf ${cond#c1_}mbit on $VICTIM" ;;
            c3_sq*) echo "  congestion: squarewave tbf ${cond#c3_sq}mbit period=120s duty=0.5 on $VICTIM" ;;
            c2_duty) echo "  congestion: organic iperf3 incast 8 streams 30s/30s on $VICTIM" ;;
        esac
    done
    exit 0
fi

# ── Preflight (real runs only) ───────────────────────────────────────────────
[[ "$(hostname -s)" == "gnn1" ]] || log "WARN: expected to run on gnn1 (got $(hostname -s))"
# shellcheck disable=SC1090
source "$HOME/dt-env.sh" 2>/dev/null || log "WARN: could not source ~/dt-env.sh"
python3 -c "import dgl" 2>/dev/null || { log "FATAL: dgl not importable (source ~/dt-env.sh)"; exit 1; }
[[ -f "$AGG" ]] || { log "FATAL: aggregator not found at $AGG"; exit 1; }
for dataset in "${D_ARR[@]}"; do
    [[ -f "${PART_CONFIGS[$dataset]}" ]] || { log "FATAL: missing part config for $dataset"; exit 1; }
done
# ip_config generated from /etc/hosts (the artifact copy is stale)
IP_CONFIG="$OUT_ROOT/ip_config.txt"
for n in $ALL_NODES; do echo "${PRIV[$n]}"; done > "$IP_CONFIG"
log "ip_config -> $IP_CONFIG: $(paste -sd' ' "$IP_CONFIG")"
# every rank reads --ip_config at this ABSOLUTE path, so it must exist on all
# nodes (missing on peers -> server crash -> DistConnectError retry hang).
for n in gnn2 gnn3 gnn4; do
    on_node "$n" "mkdir -p $OUT_ROOT" || { log "FATAL: mkdir $OUT_ROOT on $n"; exit 1; }
    scp -i "$HOME/.ssh/peerkey.pem" -o StrictHostKeyChecking=no -q \
        "$IP_CONFIG" "cc@${PRIV[$n]}:$IP_CONFIG" || { log "FATAL: ip_config -> $n"; exit 1; }
done
# iperf3 availability (only fatal if c2 requested)
IPERF_OK=1
for n in $ALL_NODES; do
    on_node "$n" 'which iperf3 >/dev/null 2>&1' || { IPERF_OK=0; log "WARN: iperf3 missing on $n"; }
done
if [[ $IPERF_OK -eq 0 && "$CONDITIONS" == *"c2"* ]]; then
    log "FATAL: c2 conditions requested but iperf3 missing (sudo apt-get install -y iperf3)"
    exit 1
fi
[[ -f "$GDY_CKPT" ]] || log "WARN: DQN checkpoint missing at $GDY_CKPT — v2 controller will use heuristic fallback"

# Optional congestion verification canary
if [[ $DO_VERIFY -eq 1 ]]; then
    log "running congestion verification canary (c1)"
    rates=$(echo "$CONDITIONS" | tr ',' '\n' | sed -n 's/^c1_\(.*\)$/\1mbit/p' | paste -sd',' -)
    if [[ -n "$rates" ]]; then
        python3 "$DIR/verify_congestion.py" --cls c1 --rates "$rates" \
            --victim "$VICTIM" --out "$OUT_ROOT/verification_c1.json" \
            || { log "FATAL: c1 verification FAILED — aborting matrix"; exit 1; }
    fi
    if [[ "$CONDITIONS" == *"c2"* ]]; then
        python3 "$DIR/verify_congestion.py" --cls c2 --victim "$VICTIM" \
            --out "$OUT_ROOT/verification_c2.json" \
            || { log "FATAL: c2 verification FAILED — aborting matrix"; exit 1; }
    fi
fi

trap 'log "INTERRUPT: tearing down"; python3 "$DIR/congestion2.py" teardown --ssh-mode peer || true' INT TERM

[[ -f "$MATRIX_TSV" ]] || echo -e "dataset\tcondition\tmethod\tseed\tstatus\tduration_s\trun_dir" > "$MATRIX_TSV"

# ── Main loop ────────────────────────────────────────────────────────────────
i=0
for row in "${PLAN_ROWS[@]}"; do
    IFS='|' read -r dataset cond method seed <<< "$row"
    i=$((i+1))
    rd="$RUNS_DIR/${dataset}__${cond}__${method}__seed${seed}"
    log "── run $i/$TOTAL: $dataset/$cond/$method/seed$seed"
    if [[ -f "$rd/p1_aggregate.json" ]]; then
        log "  SKIP (p1_aggregate.json exists)"
        continue
    fi
    mkdir -p "$rd"
    # run dir must exist on ALL nodes (each rank writes its profile locally)
    for n in gnn2 gnn3 gnn4; do on_node "$n" "mkdir -p $rd"; done

    cleanup_barrier
    rapl_barrier
    condition_start "$cond" "$rd"

    cmd=$(build_train_cmd "$method" "$dataset" "$seed" "$rd") || exit 1
    t_launch=$(date +%s)
    log "  launching: $cmd"
    timeout "$TIMEOUT_PER_RUN" python3 "$DIR/launch.py" \
        --workspace "$DIR" \
        --num_trainers 1 --num_servers 1 \
        --part_config "${PART_CONFIGS[$dataset]}" \
        --ip_config "$IP_CONFIG" \
        "$cmd" \
        --extra_envs PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 \
            NCCL_IB_DISABLE=1 OMP_NUM_THREADS=8 \
        > "$rd/run.log" 2>&1
    exit_code=$?
    t_end=$(date +%s)

    condition_stop "$rd"

    status="OK"
    [[ $exit_code -ne 0 ]] && status="FAIL(exit=$exit_code)"
    [[ $exit_code -eq 124 ]] && status="TIMEOUT"

    cat > "$rd/run_meta.json" <<EOF
{"dataset": "$dataset", "condition": "$cond", "method": "$method",
 "seed": $seed, "t_launch": $t_launch, "t_end": $t_end,
 "exit_code": $exit_code, "epochs": $EPOCHS, "batch": $BATCH,
 "victim": "$VICTIM", "cmd": "$(echo "$cmd" | sed 's/"/\\"/g')"}
EOF

    # Always attempt collection: a nonzero exit can be DGL's timing-dependent
    # shutdown abort AFTER training + profile writes completed (upstream 1.1.x
    # rpc teardown race). Classify on evidence: profiles present -> data OK.
    if collect_profiles "$rd"; then
        label=$(basename "$(ls "$rd"/*_part0_profile.json | head -1)" | sed 's/_part0_profile.json//')
        if python3 "$AGG" --run_dir "$rd" --label "$label" --warmup "$WARMUP" \
                > "$rd/aggregate.log" 2>&1; then
            [[ "$status" != "OK" ]] && { log "  NOTE: exit=$exit_code but all profiles present + aggregated -> OK_DIRTY_EXIT"; status="OK_DIRTY_EXIT"; }
        else
            status="AGG_FAIL"; log "  WARN: aggregation failed (see aggregate.log)"
        fi
    else
        [[ "$status" == "OK" ]] && status="PROFILES_MISSING"
        log "  ERROR: <4 part profiles collected — run marked $status"
    fi

    dur=$((t_end - t_launch))
    echo -e "$dataset\t$cond\t$method\t$seed\t$status\t$dur\t$rd" >> "$MATRIX_TSV"
    log "  ── $status | ${dur}s"
done

cleanup_barrier
log "matrix complete -> $MATRIX_TSV"
python3 "$DIR/parse_results.py" --matrix_dir "$OUT_ROOT" \
    --out "$OUT_ROOT/results_table.json" --md "$OUT_ROOT/results_table.md" \
    && log "results table -> $OUT_ROOT/results_table.json"
column -t -s$'\t' "$MATRIX_TSV" | tail -20
