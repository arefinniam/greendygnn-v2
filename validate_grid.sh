#!/usr/bin/env bash
# validate_grid.sh — measurement side of the sim-to-real validation grid.
# HOLDOUT cells (never used in fitting): W in {4,8,32} x {c1_200mbit, c1_50mbit,
# c4_10ms} per dataset. Combined with the calibration runs (clean x all W,
# W16 x all severities) this yields the predicted-vs-measured error grid that
# replaces the synthesized Fig 8. Reuses calibrate_cluster.sh's one_run pattern.
# Usage: bash validate_grid.sh <dataset> <part_config> <n_classes>
set -o pipefail
# node registry — extend here for >4 nodes (audit M2)
ALL_IPS="10.52.2.119 10.52.3.217 10.52.3.123 10.52.3.89"
PEER_IPS="10.52.3.217 10.52.3.123 10.52.3.89"
WS="$HOME/gdy2"; cd "$WS"
DATASET="${1:?}"; PART_CONFIG="${2:?}"; NCLASSES="${3:?}"
BATCH=2000; EPOCHS=6; SEED=0
CAL="$WS/calib_runs"; mkdir -p "$CAL"
source "$HOME/dt-env.sh" >/dev/null 2>&1
EXTRA_ENVS="LD_LIBRARY_PATH=${LD_LIBRARY_PATH} PATH=${PATH} NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=6"
TRAIN_COMMON="--num_gpus 1 --num_hidden 16 --num_layers 2 --fan_out 10,25 --lr 0.003 --dropout 0.5 --log_every 20"

rapl_barrier() {
  for ip in $ALL_IPS; do
    ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no cc@"$ip" \
      'sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/name /sys/class/powercap/intel-rapl:*/max_energy_range_uj 2>/dev/null' >/dev/null 2>&1
  done
}
cleanup_procs() {
  for ip in $ALL_IPS; do
    ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no cc@"$ip" \
      'pkill -9 -f "train_rapidgn[n]|launch.p[y]" 2>/dev/null; fuser -k 30050/tcp 30051/tcp 29500/tcp 2>/dev/null; true' >/dev/null 2>&1
  done
  sleep 3
}
one_run() {
  local tag="$1" W="$2"; local OUT="$CAL/$tag"
  if ls "$OUT"/*_part0_profile.json >/dev/null 2>&1; then echo "SKIP $tag"; return 0; fi
  mkdir -p "$OUT"
  for ip in $PEER_IPS; do
    ssh -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no cc@"$ip" "mkdir -p $OUT" >/dev/null 2>&1
  done
  cleanup_procs; rapl_barrier
  local UDF="python3 $WS/train_rapidgnn.py --graph_name $DATASET --ip_config $WS/ip_config.txt --part_config $PART_CONFIG --num_epochs $EPOCHS --batch_size $BATCH --out_dir $OUT --n_classes $NCLASSES --seed $SEED --window_size $W --cache_size 100000 $TRAIN_COMMON"
  echo "=== RUN $tag ==="
  timeout 1800 python3 "$WS/launch.py" --workspace "$WS" \
    --num_trainers 1 --num_servers 1 --num_samplers 0 \
    --part_config "$PART_CONFIG" --ip_config "ip_config.txt" \
    "$UDF" --extra_envs $EXTRA_ENVS > "$OUT/run.log" 2>&1
  echo "rc=$? $tag"
  for a in 1 2 3 4 5; do
    for ip in $PEER_IPS; do
      scp -i ~/.ssh/peerkey.pem -o StrictHostKeyChecking=no -q "cc@$ip:$OUT/*_profile.json" "$OUT/" 2>/dev/null || true
    done
    [ "$(ls "$OUT"/*_profile.json 2>/dev/null | wc -l)" -ge 4 ] && { echo "collected $tag"; return 0; }
    sleep 8
  done
  echo "WARN: <4 profiles $tag"
}

for COND in c1_200mbit c1_50mbit c4_10ms; do
  case "$COND" in
    c1_*) python3 "$WS/congestion2.py" apply --cls c1 --victims gnn4 --rate "${COND#c1_}" \
            --ssh-mode peer --journal "$CAL/${DATASET}_vg_${COND}.jsonl" || exit 1 ;;
    c4_*) d="${COND#c4_}"; d="${d%ms}"
          python3 "$WS/congestion2.py" apply --cls c4 --victims gnn4 --delay "$d" --jitter 0 --loss 0 \
            --ssh-mode peer --journal "$CAL/${DATASET}_vg_${COND}.jsonl" || exit 1 ;;
  esac
  for W in 4 8 32; do
    one_run "${DATASET}_vg_${COND}_W${W}" "$W"
  done
  python3 "$WS/congestion2.py" teardown --ssh-mode peer || exit 1
done
cleanup_procs
echo "##### validation grid done ($DATASET) #####"
