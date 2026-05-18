# PolyPPO Next Novelty Result

Status: completed with a negative quality-gated result on commit `6008091cb89af1ad741287c893c927a63047b173`.

## Objective

The follow-up hypothesis was:

> Naive diversity rewards can push VQ-BeT off-manifold and hurt pass@1, but quality-gated / return-conditioned diversity can preserve the pass@k coverage gain while recovering pass@1 and robustness.

This run tested the smallest valid version: PPO over RVQ code-id tuples only. Continuous actions remain deterministic VQ-BeT decoder/head outputs conditioned on the sampled or replayed code IDs.

## Readiness And Gates

- Latest PR review-gate fixes were pushed and resolved, including the final image-stacking and fake grouped-pass@k stress-eval fixes.
- Focused suite after the quality-gated implementation and final review fixes: `62 passed, 2 warnings`.
- GPU rule followed: at least three clean polls before each CUDA job; after a user request to free a GPU, all remaining work was run on GPU 0 only.
- Quality-gated train artifact has `git_dirty=false`.
- Grouped pass@k and paired eval artifacts validated with `validation_status=passed`.
- Paired 100-episode seed manifests match across Direct, PPO, naive PolyPPO, and quality-gated PolyPPO.

## Quality-Gated Method

Config: `configs/polyppo/pusht_novelty_polyppo_qg_code_32x8.yaml`

Objective:

```text
diversity_kind: quality_bad_code
quality_gate: quantile
quality_quantile: 0.75
lambda_div: 0.03
lambda_bad: 0.03
```

Training artifact: `outputs/polyppo/pusht_novelty_polyppo_qg_code_32x8/polyppo_train_info.json`

| Field | Value |
| --- | ---: |
| Updates | 3 |
| PPO ratio max abs error before update | 0.0 |
| Checkpoint saved | true |
| Loss finite | true |
| Nonzero gradients | true |
| Parameters changed | true |
| Checkpoint hash | `53d3a50a589a52d40d096480ca6920c2bb54cb8b93d609ecab6768de99896c2c` |

## Grouped Same-Start Pass@k

Config: `configs/polyppo/pusht_grouped_passk_novelty_full.yaml`

Artifact: `outputs/polyppo/pusht_grouped_passk_novelty_full/grouped_passk_summary.json`

Setup: 8 restored PushT starts, 16 attempts per start, same start seeds across all methods.

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | coverage@16 | Avg max overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct VQ-BeT | 0.250 | 0.500 | 0.750 | 0.875 | 0.875 | 7/8 | 0.7916 |
| PPO no diversity | 0.375 | 0.625 | 0.750 | 0.875 | 0.875 | 7/8 | 0.7954 |
| PolyPPO code diversity | 0.250 | 0.625 | 0.750 | 0.875 | 0.875 | 7/8 | 0.7907 |
| Quality-gated PolyPPO code | 0.250 | 0.625 | 0.750 | 0.875 | 0.875 | 7/8 | 0.7907 |

Result: quality-gated code diversity matched naive code diversity and did not beat PPO or Direct on grouped coverage, pass@k, or average max overlap.

## Paired 100-Episode Eval

Config: `configs/polyppo/pusht_novelty_paired_eval_100.yaml`

Artifact: `outputs/polyppo/pusht_novelty_paired_eval_100/eval_polyppo_summary.json`

Setup: 100 standard PushT episodes, seed block `180000`, same seed manifest across all methods.

| Method | pass@1 | Success CI95 | Avg max overlap |
| --- | ---: | --- | ---: |
| Direct VQ-BeT | 0.440 | [0.3422, 0.5378] | 0.7955 |
| PPO no diversity | 0.440 | [0.3422, 0.5378] | 0.7955 |
| PolyPPO code diversity | 0.430 | [0.3325, 0.5275] | 0.7954 |
| Quality-gated PolyPPO code | 0.430 | [0.3325, 0.5275] | 0.7954 |

Result: quality-gated PolyPPO did not recover pass@1. It tied naive PolyPPO and underperformed Direct/PPO by one success in 100 paired episodes.

## Decision

No publishable novelty was found for quality-gated code diversity on this evidence ladder. The kill criterion is met: the new method does not beat PPO-without-diversity or inference-only Direct VQ-BeT on pass@1, grouped pass@k, coverage, robustness proxy, or overlap at matched compute.

Do not scale this quality-gated code-diversity variant to 500+ episodes. The next meaningful research step is not a larger blind run of the same objective. If the project continues, prioritize a genuinely different mechanism:

1. PolyDistill / top-of-set supervised CE from grouped successful attempts.
2. Action or endpoint trajectory diversity instead of raw RVQ-code diversity.
3. Success-only BC or KL-to-base regularization with verified base log-probs.
4. A second environment after a positive PushT signal, not before.

## Commands

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q tests/test_pusht_state_restore.py tests/test_vqbet_code_ppo.py tests/test_polyppo_scaffold.py tests/test_artifact_validator.py tests/test_run_registry.py tests/test_polyppo_metrics.py tests/test_polyppo_rollout_schema.py tests/test_polyppo_trainer_smoke.py tests/test_polyppo_scripts.py
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q tests/test_pusht_state_restore.py tests/test_vqbet_code_ppo.py tests/test_polyppo_scaffold.py tests/test_artifact_validator.py tests/test_run_registry.py tests/test_polyppo_metrics.py tests/test_polyppo_rollout_schema.py tests/test_polyppo_trainer_smoke.py tests/test_polyppo_scripts.py tests/test_eval_helpers.py
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -u -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_novelty_polyppo_qg_code_32x8.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -u -m lerobot.scripts.eval_grouped_passk --config configs/polyppo/pusht_grouped_passk_novelty_full.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -u -m lerobot.scripts.eval_polyppo_checkpoints --config configs/polyppo/pusht_novelty_paired_eval_100.yaml
```
