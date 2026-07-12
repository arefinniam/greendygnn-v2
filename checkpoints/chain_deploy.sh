#!/bin/bash
# Wait for the synthetic machinery-validation run (pid $1), then train the
# DEPLOY checkpoint on BOTH real calibrations with the updated simulator.
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[chain] synthetic run finished at $(date)" 
cd ~/greendygnn_work/code
python3 train_agent.py \
  --calib data/calib_reddit.json,data/calib_ogbn-products.json \
  --out checkpoints/dqn_v2_real.pt \
  --episodes 24000 --update-every 2 --eval-episodes 500 --seed 0 \
  > checkpoints/train_real.log 2>&1
echo "[chain] deploy training finished at $(date), rc=$?"
tail -40 checkpoints/train_real.log
