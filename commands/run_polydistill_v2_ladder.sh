#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="/tmp/gymnasium_vendor:/tmp/termcolor_pkg:${PYTHONPATH:-}"
LOG_DIR="/pub7/neel2/polyppo_runs/logs"
mkdir -p "$LOG_DIR"

echo "[polyDistill-v2] repo: $ROOT"
echo "[polyDistill-v2] commit: $(git rev-parse HEAD)"
echo "[polyDistill-v2] status:"
git status --short --branch

echo "[polyDistill-v2] focused tests"
pytest -q \
  tests/test_polydistill.py \
  tests/test_artifact_validator.py \
  tests/test_vqbet_code_ppo.py \
  tests/test_pusht_state_restore.py \
  tests/test_eval_helpers.py

echo "[polyDistill-v2] preflight GPU polls"
for i in 1 2 3; do
  echo "poll $i"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
  if [ "$i" -lt 3 ]; then sleep 30; fi
done

echo "[polyDistill-v2] smoke collection on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.collect_polydistill_v2_dataset \
  --config configs/polyppo/pusht_polydistill_v2_collect_smoke_8x16_top4.yaml \
  2>&1 | tee "$LOG_DIR/polydistill_v2_collect_smoke_8x16_top4.log"

echo "[polyDistill-v2] main train collection on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.collect_polydistill_v2_dataset \
  --config configs/polyppo/pusht_polydistill_v2_collect_train_64x16_top4.yaml \
  2>&1 | tee "$LOG_DIR/polydistill_v2_collect_train_64x16_top4.log"

echo "[polyDistill-v2] start validation collection on GPU 0 and training sweep on GPU 1"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.collect_polydistill_v2_dataset \
  --config configs/polyppo/pusht_polydistill_v2_collect_val_32x16_top4.yaml \
  >"$LOG_DIR/polydistill_v2_collect_val_32x16_top4.log" 2>&1 &
VAL_PID=$!

(
  set -euo pipefail
  for cfg in \
    configs/polyppo/pusht_polydistill_v2_train_top1_ce_64x16.yaml \
    configs/polyppo/pusht_polydistill_v2_train_topm4_tau007_kl001_64x16.yaml \
    configs/polyppo/pusht_polydistill_v2_train_success_soft_m4_tau007_kl001_64x16.yaml \
    configs/polyppo/pusht_polydistill_v2_train_adv_m4_tau05_kl001_64x16.yaml
  do
    name="$(basename "$cfg" .yaml)"
    echo "[polyDistill-v2] train $name on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python -u -m lerobot.scripts.train_polydistill_v2 --config "$cfg" \
      2>&1 | tee "$LOG_DIR/${name}.log"
  done
) &
TRAIN_PID=$!

wait "$VAL_PID"
wait "$TRAIN_PID"

echo "[polyDistill-v2] held-out grouped pass@k on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.eval_grouped_passk \
  --config configs/polyppo/pusht_polydistill_v2_grouped_val_32x16.yaml \
  2>&1 | tee "$LOG_DIR/polydistill_v2_grouped_val_32x16.log"

echo "[polyDistill-v2] paired 100 standard eval on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.eval_polyppo_checkpoints \
  --config configs/polyppo/pusht_polydistill_v2_paired_eval_100.yaml \
  2>&1 | tee "$LOG_DIR/polydistill_v2_paired_eval_100.log"

echo "[polyDistill-v2] stress eval on GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u -m lerobot.scripts.eval_polyppo_checkpoints \
  --config configs/polyppo/pusht_polydistill_v2_stress_eval_100.yaml \
  2>&1 | tee "$LOG_DIR/polydistill_v2_stress_eval_100.log"

echo "[polyDistill-v2] completed ladder; summarize into reports/polyppo_polydistill_v2_results.md"

