# PolyDistill v2 Coverage-Compression Status

Date: 2026-05-20
Branch: `codex/polyppo-foundation`
PR: `https://github.com/sabdulmajid/bid_lerobot/pull/1`

## Status

Blocked before GPU experiment launch by another active two-GPU job.

Repository and PR hygiene are complete:

- Local branch: `codex/polyppo-foundation`
- Remote PR branch: pushed and synced; verify exact current head with `git rev-parse HEAD origin/codex/polyppo-foundation`.
- Commit author/committer identity: `sabdulmajid <ayman.hasib@outlook.com>`
- Working tree was clean immediately after the pushed v2 implementation commit.

Implemented and pushed:

- Goal file: `reports/polyppo_polydistill_v2_goal.md`
- V2 module: `lerobot/common/polyppo/distill_v2.py`
- Collection CLI: `lerobot/scripts/collect_polydistill_v2_dataset.py`
- Training CLI: `lerobot/scripts/train_polydistill_v2.py`
- Resume ladder: `commands/run_polydistill_v2_ladder.sh`
- V2 configs under `configs/polyppo/`
- V2 unit tests in `tests/test_polydistill.py`

## Tests

Focused command:

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q \
  tests/test_polydistill.py \
  tests/test_artifact_validator.py \
  tests/test_vqbet_code_ppo.py \
  tests/test_pusht_state_restore.py \
  tests/test_eval_helpers.py
```

Result:

```text
37 passed, 2 warnings
```

`py_compile` also passed for the new v2 module and CLIs.

## Exact Blocker

Both GPUs were occupied by an unrelated DS1000 evaluation process during the launch window:

```text
PID 2497686
python scripts/run_ds1000_eval.py \
  --model-path openai/gpt-oss-20b \
  --adapter-path artifacts/checkpoints/sft/gpt_oss_20b_sft_v3_clean_imports_1024/checkpoint-25 \
  --config-path configs/training/sft_gpt_oss_20b_lora_v3_clean_imports_1024.yaml \
  --libraries Pytorch Numpy Sklearn Scipy \
  --per-library 25 \
  --require-reference-pass \
  --output artifacts/eval/ds1000_v2/ds1000_sft_v3_step25_ml_refpass_raw_100_v1.json \
  --max-new-tokens 256 \
  --seed 1337 \
  --execution-timeout-sec 60
```

Repeated polls showed GPU utilization and large allocations on both devices:

```text
GPU 0: about 30.7 GiB allocated, 32-38% utilization
GPU 1: about 28.7 GiB allocated, 51-55% utilization
```

No PolyDistill v2 PushT GPU jobs were launched, to avoid contaminating both experiments.

## Resume Command

When both GPUs are free, run:

```bash
cd /tmp/bid_lerobot_polyppo_review
git fetch origin --prune
git checkout codex/polyppo-foundation
git pull --ff-only
bash commands/run_polydistill_v2_ladder.sh
```

The ladder will:

1. Re-run the focused test gate.
2. Poll GPUs three times.
3. Collect smoke `8x16`, train `64x16`, and held-out validation `32x16` datasets.
4. Train four v2 objectives on GPU 1.
5. Run held-out grouped pass@k, paired standard 100, and stress eval on GPU 0.

## Expected Output Paths

- Smoke dataset: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_smoke_8x16_top4`
- Train dataset: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_train_dataset_64x16_top4`
- Validation dataset: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_val_dataset_32x16_top4`
- Training runs: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_*_64x16`
- Grouped validation: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_grouped_val_32x16`
- Paired eval: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_paired_eval_100`
- Stress eval: `/pub7/neel2/polyppo_runs/pusht_polydistill_v2_stress_eval_100`
- Logs: `/pub7/neel2/polyppo_runs/logs`

## Acceptance Criteria Not Yet Evaluated

No v2 variant has been evaluated yet, so none of the goal acceptance criteria can be claimed.
