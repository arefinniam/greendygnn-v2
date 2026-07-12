#!/usr/bin/env bash
###############################################################################
# calibrate_cluster.sh — Phase 1 (clean W-sweep) + Phase 2 (congestion axes)
# of CALIBRATION_PROTOCOL.md. Runs ON gnn1. Static-W runs via train_rapidgnn.py
# (no controller involved). Profiles land in ~/gdy2/calib_runs/<tag>/ on each
# node; collect + fit happens off-cluster (fit_calibration.py).
#
# Usage: bash calibrate_cluster.sh <dataset> <part_config> <n_classes> [phase]
#   phase: 1 = W-sweep only, 2 = congestion axes only, all = both (default)
###############################################################################
set -o pipefail
# node registry — extend here for >4 nodes (audit M2)
ALL_IPS="10.52.2.119 10.52.3.217 10.52.3.123 10.52.3.89"
PEER_IPS="10.52.3.217 10.52.3.123 10.52.3.89"
WS="$HOME/gdy2"
cd "$WS"
DATASET="${1:?dataset}"; PART_CONFIG="${2:?part_config}"; NCLASSES="${3:?n_classes}"
PHASE="${4:-all}"
BATCH=2000; EPOCHS=6; SEED=0
WLIST="1 2 4 8 16 32 64 128"
RATES="1000mbit 500mbit 200mbit 100mbit 50mbit"
DELAYS="5 10 20"
VICTIM=gnn4
CAL="$WS/calib_runs"
mkdir -p "$CAL"

source "$HOME/dt-env.sh" >/dev/null 2>&1
EXTRA_ENVS="LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PATH=${PATH} NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=6"
TRAIN_COMMON="--num_gpus 1 --num_hidden 16 --num_layers 2 --fan_out 10,25 --lr 0.003 --dropout 0.5 --log_every 20"

rapl_barrier() {
  for ip in $ALL_IPS; do
    ok=0
    for try in 1 2 3; do
      res=$(ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no -o ConnectTimeout=8 cc@"$ip" \
        'sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/name /sys/class/powercap/intel-rapl:*/max_energy_range_uj 2>/dev/null; for f in /sys/class/powercap/intel-rapl:[0-9]*/energy_uj; do test -r "$f" || { echo NOPE; exit 0; }; done; echo ALLOK' 2>/dev/null)
      [ "$res" = "ALLOK" ] && { ok=1; break; }; sleep 1
    done
    [ "$ok" = "1" ] || { echo "FATAL: RAPL not readable on $ip"; exit 3; }
  done
}

cleanup_procs() {
  for ip in $ALL_IPS; do
    ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no cc@"$ip" \
      'pkill -9 -f "train_rapidgn[n]|train_greendygn[n]|train_defaul[t]|launch.p[y]" 2>/dev/null; fuser -k 30050/tcp 30051/tcp 30052/tcp 30053/tcp 29500/tcp 29501/tcp 2>/dev/null; true' >/dev/null 2>&1
  done
  sleep 3
}

one_run() {  # one_run <tag> <window_size> [extra...]
  local tag="$1" W="$2"; shift 2
  local OUT="$CAL/$tag"
  # resumable: skip if part0 profile already exists with steps
  if ls "$OUT"/*_part0_profile.json >/dev/null 2>&1; then echo "SKIP $tag (exists)"; return 0; fi
  mkdir -p "$OUT"
  for ip in $PEER_IPS; do
    ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no cc@"$ip" "mkdir -p $OUT" >/dev/null 2>&1
  done
  cleanup_procs; rapl_barrier
  local UDF="python3 $WS/train_rapidgnn.py --graph_name $DATASET --ip_config $WS/ip_config.txt --part_config $PART_CONFIG --num_epochs $EPOCHS --batch_size $BATCH --out_dir $OUT --n_classes $NCLASSES --seed $SEED --window_size $W --cache_size 100000 $TRAIN_COMMON $*"
  echo "=== RUN $tag (W=$W) ==="
  timeout 1800 python3 "$WS/launch.py" --workspace "$WS" \
    --num_trainers 1 --num_servers 1 --num_samplers 0 \
    --part_config "$PART_CONFIG" --ip_config "ip_config.txt" \
    "$UDF" --extra_envs $EXTRA_ENVS > "$OUT/run.log" 2>&1
  local rc=$?
  echo "rc=$rc $tag"
  # collect peer profiles with retry
  for attempt in 1 2 3 4 5; do
    for ip in $PEER_IPS; do
      scp -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no -q "cc@$ip:$OUT/*_profile.json" "$OUT/" 2>/dev/null || true
    done
    local c; c=$(ls "$OUT"/*_profile.json 2>/dev/null | wc -l)
    [ "$c" -ge 4 ] && { echo "collected 4 profiles for $tag"; return 0; }
    sleep 8
  done
  echo "WARN: <4 profiles for $tag"
}

if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
  echo "##### PHASE 1: clean W-sweep ($DATASET) #####"
  python3 "$WS/congestion2.py" teardown --ssh-mode peer || exit 1
  for W in $WLIST; do
    one_run "${DATASET}_clean_W${W}" "$W"
  done
fi

if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
  echo "##### PHASE 2: congestion axes ($DATASET, victim=$VICTIM, W=16) #####"
  for RATE in $RATES; do
    python3 "$WS/congestion2.py" apply --cls c1 --victims "$VICTIM" --rate "$RATE" \
      --ssh-mode peer --journal "$CAL/${DATASET}_c1_${RATE}.jsonl" || exit 1
    one_run "${DATASET}_c1_${RATE}_W16" 16
    python3 "$WS/congestion2.py" teardown --ssh-mode peer || exit 1
  done
  for D in $DELAYS; do
    python3 "$WS/congestion2.py" apply --cls c4 --victims "$VICTIM" --delay "$D" --jitter 0 --loss 0 \
      --ssh-mode peer --journal "$CAL/${DATASET}_c4_${D}ms.jsonl" || exit 1
    one_run "${DATASET}_c4_${D}ms_W16" 16
    python3 "$WS/congestion2.py" teardown --ssh-mode peer || exit 1
  done
fi

cleanup_procs
python3 "$WS/congestion2.py" teardown --ssh-mode peer
echo "##### calibration runs done: $CAL #####"
