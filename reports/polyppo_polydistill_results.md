# PolyDistill Top-of-Set CE Results

Date: 2026-05-18
Branch: `codex/polyppo-foundation`
Implementation commit: `c9b4935ea2ce11e5ab1f246690ccc3450043b268`

## Objective

Implement PolyDistill/top-of-set cross entropy:

1. Sample many VQ-BeT candidates from the same PushT start states.
2. Select the best attempt per start, preferring success, then maximum overlap, then return.
3. Distill the selected RVQ code-id sequence back into the VQ-BeT code policy.
4. Evaluate whether pass@k evidence converts into better pass@1 behavior.

## Artifacts

- Dataset config: `configs/polyppo/pusht_polydistill_collect_8x16.yaml`
- Train config: `configs/polyppo/pusht_polydistill_train_8x16.yaml`
- Grouped pass@k config: `configs/polyppo/pusht_polydistill_grouped_passk_8x16.yaml`
- Paired eval config: `configs/polyppo/pusht_polydistill_paired_eval_100.yaml`
- Dataset artifact: `outputs/polyppo/pusht_polydistill_dataset_8x16/polydistill_dataset.json`
- Train artifact: `outputs/polyppo/pusht_polydistill_topset_8x16/polydistill_train_info.json`
- Grouped pass@k artifact: `outputs/polyppo/pusht_polydistill_grouped_passk_8x16/grouped_passk_summary.json`
- Paired eval artifact: `outputs/polyppo/pusht_polydistill_paired_eval_100/eval_polyppo_summary.json`

## Collection Result

The collector sampled 8 PushT start states with 16 attempts per start, for 128 candidate episodes. Selection used success first, then max overlap, then return.

| Metric | Value |
|---|---:|
| Candidate success rate | 0.390625 |
| Selected success rate | 0.875 |
| Selected average max overlap | 0.964801 |
| Selected average return | 83.005905 |
| Selected valid code steps | 2051 |

This validates the motivating premise: pass@16 contains much stronger candidates than a random single sample.

## Training Result

Training used CE on selected RVQ code IDs and updated only the VQ-BeT code-logit head.

| Check | Result |
|---|---:|
| Checkpoint saved | true |
| Loss finite | true |
| Nonzero gradients | true |
| Parameters changed | true |
| Valid selected steps | true |

CE losses by epoch were approximately `2.640`, `2.651`, `2.655`, `2.622`, `2.639`. The loss was finite and trainable, but it did not show a strong monotonic decrease in this small 5-epoch run.

## Grouped Restored-Start pass@k

Evaluation used the same 8 restored starts and 16 attempts per start for each method. Seed manifests match across methods.

| Method | Success rate over attempts | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | coverage@16 | Avg max overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_vqbet | 0.5391 | 0.250 | 0.500 | 0.750 | 0.875 | 0.875 | 7 | 0.7916 |
| ppo_no_diversity | 0.5156 | 0.375 | 0.625 | 0.750 | 0.875 | 0.875 | 7 | 0.7954 |
| polyppo_code_diversity | 0.5234 | 0.250 | 0.625 | 0.750 | 0.875 | 0.875 | 7 | 0.7907 |
| polyppo_quality_gated_code | 0.5234 | 0.250 | 0.625 | 0.750 | 0.875 | 0.875 | 7 | 0.7907 |
| polydistill_topset_8x16 | 0.4531 | 0.500 | 0.625 | 0.750 | 0.875 | 0.875 | 7 | 0.7926 |

PolyDistill improves grouped pass@1 to `0.500`, beating direct VQ-BeT at `0.250` and PPO without diversity at `0.375`. It does not improve pass@16 or coverage, and it lowers the total success rate across all stochastic attempts.

## Paired 100-Episode Standard PushT Eval

Evaluation used the same 100 seeds for all methods on standard PushT.

| Method | pass@1 / success rate | Avg max overlap |
|---|---:|---:|
| direct_vqbet | 0.44 | 0.795541 |
| ppo_no_diversity_32x8 | 0.44 | 0.795512 |
| polyppo_code_diversity_32x8 | 0.43 | 0.795408 |
| polyppo_quality_gated_code_32x8 | 0.43 | 0.795406 |
| polydistill_topset_8x16 | 0.42 | 0.768697 |

On the ordinary 100-episode distribution, this first PolyDistill checkpoint does not beat the direct or PPO baselines.

## Interpretation

The result is mixed but informative. The selector found substantially better trajectories inside pass@16 sampling, and CE distillation changed the first-sample behavior on the fixed restored-start benchmark. That is the positive signal.

The negative signal is that the same checkpoint does not generalize to standard 100-episode PushT. The likely reasons are:

- The distillation dataset is too small: 8 start states and 2051 selected steps can overfit the sampled starts.
- Hard top-1 CE can collapse probability mass toward a lucky code sequence instead of preserving robust alternatives.
- The policy updates only RVQ code logits; continuous offsets remain deterministic outputs conditioned on features and sampled codes.
- The selected candidates are high-return under stochastic sampling, but not necessarily globally consistent action-code supervision.
- There is no KL-to-base or BC regularization in this first PolyDistill run, so small logits shifts can hurt the broader state distribution.
- The training loss did not decrease strongly, suggesting the code-head-only target may be underpowered or noisy at this scale.

## Decision

Do not claim PolyDistill as a solved win from this run. The publishable statement is narrower:

> Top-of-set CE can compress pass@k evidence into better first-attempt behavior on matched restored starts, but the current small 8x16 run does not yet improve standard PushT pass@1.

## Recommended Next Step

Do not scale the same exact config blindly. The next meaningful experiment should use a larger train/validation split:

1. Collect at least 64 to 128 start states with 16 attempts each.
2. Hold out start states for grouped pass@k validation.
3. Distill from top-1 or top-m successful attempts with KL-to-base regularization.
4. Early-stop on held-out grouped pass@1, not training CE.
5. Re-run standard paired eval only if held-out grouped pass@1 improves.

