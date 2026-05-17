# PolyPPO Novelty Search

Status: prepared, blocked before GPU execution by active GPU use.

## Objective

Run the novelty-discovery loop from `reports/polyppo_novelty_discovery_goal.md`:

- verify PR correctness state;
- rerun focused tests;
- reproduce baseline / sampler evidence;
- implement real grouped pass@k from repeated same-start attempts;
- collect a 32x8 rollout batch;
- run at least one multi-update PPO and one multi-update PolyPPO config;
- evaluate at least 100 paired episodes for top PPO vs top PolyPPO;
- report numeric novelty evidence or a rigorous negative result.

## CPU-Only Progress

- Restored the actual novelty goal file into the active PR branch at `reports/polyppo_novelty_discovery_goal.md`.
- Added multi-update support to `train_polyppo_one_update` through `train.num_updates`; the existing one-update path remains the default.
- Added a grouped pass@k evaluator at `lerobot/scripts/eval_grouped_passk.py`. It snapshots each PushT start state, restores the exact same start for each attempt, records per-attempt success/max overlap, and rejects `pass@k` requests larger than `attempts_per_start`.
- Added 32x8 rollout and novelty train/eval configs:
  - `configs/polyppo/pusht_rollout_32x8.yaml`
  - `configs/polyppo/pusht_novelty_ppo_32x8.yaml`
  - `configs/polyppo/pusht_novelty_polyppo_code_32x8.yaml`
  - `configs/polyppo/pusht_grouped_passk_novelty.yaml`
- Added a mock regression covering `train.num_updates > 1`.

## Verification So Far

Pre-wait focused tests:

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q \
  tests/test_polyppo_trainer_smoke.py \
  tests/test_polyppo_scripts.py \
  tests/test_eval_helpers.py \
  tests/test_polyppo_metrics.py \
  tests/test_polyppo_utils.py \
  tests/test_polyppo_rollout_schema.py \
  tests/test_vqbet_code_ppo.py \
  tests/test_artifact_validator.py \
  tests/test_pusht_state_restore.py \
  tests/test_run_registry.py \
  tests/test_policy_factory.py \
  tests/test_sampler_return_contracts.py
```

Result: `44 passed, 2 warnings`.

The full novelty acceptance suite has not been rerun after this patch because the user reported another task using the GPUs and requested waiting.

## Current Blocker

The novelty goal requires GPU rollout/training/evaluation. Both GPUs are currently occupied by another Python process, and the user explicitly requested waiting. No 32x8 rollout, multi-update PPO, multi-update PolyPPO, grouped pass@k eval, or 100-paired-episode novelty result has been executed yet.

Observed GPU occupancy:

```text
GPU 0: python pid 3694298, ~29 GiB
GPU 1: python pid 3694298, ~28 GiB
```

## Resume Commands

Once GPUs are free, run:

```bash
cd /tmp/bid_lerobot_polyppo_review

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q \
  tests/test_policy_factory.py \
  tests/test_eval_helpers.py \
  tests/test_polyppo_scripts.py \
  tests/test_sampler_return_contracts.py \
  tests/test_vqbet_code_ppo.py \
  tests/test_polyppo_scaffold.py \
  tests/test_artifact_validator.py \
  tests/test_polyppo_trainer_smoke.py \
  tests/test_polyppo_utils.py \
  tests/test_polyppo_rollout_schema.py \
  tests/test_polyppo_metrics.py \
  tests/test_pusht_state_restore.py \
  tests/test_run_registry.py

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.collect_polyppo_rollouts --config configs/polyppo/pusht_rollout_32x8.yaml

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_novelty_ppo_32x8.yaml

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_novelty_polyppo_code_32x8.yaml

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.eval_grouped_passk --config configs/polyppo/pusht_grouped_passk_novelty.yaml
```

Then update `reports/polyppo_novelty_search.md` and `reports/polyppo_novelty_summary.json` with numeric results and promotion/kill decisions.
