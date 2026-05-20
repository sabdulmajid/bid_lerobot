# PolyDistill v2 / Coverage-Compression Goal

You are in `sabdulmajid/bid_lerobot`, PR #1 branch `codex/polyppo-foundation`.

Mission: turn the current mixed PolyPPO/PolyDistill evidence into a serious novelty-search experiment. The discovered signal is: many VQ-BeT continuations from the same PushT start contain good solutions, but naive PolyPPO/QG code-diversity and small top-of-set CE do not generalize to ordinary pass@1. Treat this as a coverage-compression problem, not a failure.

## Current evidence to preserve
- PPO/PolyPPO foundation exists and tests pass.
- QG PolyPPO: paired pass@1 Direct 0.44, PPO 0.44, naive PolyPPO 0.43, QG 0.43; grouped coverage@16 all 7/8.
- PolyDistill v1: 8 starts x 16 attempts; candidate success 0.390625; selected top success 0.875; grouped restored-start pass@1 Direct 0.25, PPO 0.375, PolyDistill 0.50; standard paired 100 pass@1 Direct/PPO 0.44, PolyDistill 0.42.

## First: repository / PR hygiene
1. Verify whether commits `c9b4935` and `c8d61c3` are actually on the remote PR. If local branch is ahead, push them; if HTTPS auth blocks, use `gh auth status`, `gh auth login`, or report the exact credential blocker and create a patch file.
2. Do not merge until the branch is pushed, clean, and tests pass.
3. Keep all author/committer identity as `sabdulmajid <ayman.hasib@outlook.com>`.

## Use both GPUs
- GPU 0: candidate collection, grouped validation/eval, baseline eval.
- GPU 1: PolyDistill training sweeps.
- Log `nvidia-smi`, CUDA device names, GPU id per run, git SHA, dirty flag, command, checkpoint hash, seed manifest, wall time, and output path.

## Core hypothesis
Best-of-k/top-of-set candidates expose latent coverage in VQ-BeT. Distillation failed to generalize because v1 used only 8 starts and hard top-1 labels. Test whether larger data, held-out grouped validation, soft top-m targets, KL-to-base, and early stopping can convert pass@k coverage into pass@1.

## Implement PolyDistill v2
Add/extend scripts/configs as needed:
- `collect_polydistill_dataset.py`
- `train_polydistill.py`
- `eval_polydistill.py` or existing eval scripts
- configs under `configs/polyppo/`
- report `reports/polyppo_polydistill_v2_results.md`
- summary `reports/polyppo_polydistill_v2_summary.json`

### Dataset collection
Collect multiple datasets:
1. Smoke: 8 starts x 16 attempts.
2. Main train: 64 starts x 16 attempts minimum; stretch 128 x 16 or 64 x 32.
3. Held-out grouped validation: 32 starts x 16 attempts, start seeds disjoint from train.
4. Optional stress train: observation/action noise metadata, but do not mix unless marked.

For each attempt store: start_id, seed, prefix/env state hash, attempt_id, code ids, action chunks, rewards, success, max overlap, episode length, final return, validity mask, checkpoint hash, sampler temp, and observation/action summaries.

### Distillation objectives to try
Run a small but real sweep. Do not only repeat hard top-1 CE.

Methods:
1. `top1_ce`: CE on best successful/top-overlap code sequence.
2. `topm_soft_ce`: top-m attempts weighted by softmax(max_overlap / tau), m in {2,4,8}, tau in {0.03,0.07,0.15}.
3. `success_filtered_soft_ce`: weight only successful attempts; if none, top-overlap fallback.
4. `adv_weighted_bc`: all attempts weighted by normalized return/overlap advantage within start set.
5. `pairwise_pref` if feasible: DPO/BT-style selected-vs-bad code preference within each start.

Regularizers:
- KL-to-base or log-prob anchor to pretrained VQ-BeT.
- entropy floor if collapse appears.
- optional mix with original/base behavior loss if available.

Sweep ranges:
- learning rate: 1e-5, 3e-5, 1e-4
- KL weight: 0.0, 0.01, 0.05, 0.1
- train epochs: 1, 3, 5 with early stopping on held-out grouped pass@1/avg overlap
- freeze vision/trunk; train RVQ/code head first. Only unfreeze more if underfitting.

## Evaluation protocol
Every candidate checkpoint gets:
1. Held-out grouped same-start pass@k on validation starts: pass@1/2/4/8/16, coverage@16, avg max overlap.
2. Standard paired PushT eval: 100 episodes first.
3. If promising, standard paired 500 episodes.
4. Stress eval: observation noise and action noise, at least 100 paired episodes for promising configs.

Baselines:
- Direct pretrained VQ-BeT.
- PPO no-diversity.
- naive PolyPPO/QG PolyPPO if checkpoints exist.
- PolyDistill v1.
- New PolyDistill v2 variants.

Important: also run direct VQ-BeT grouped pass@k with the same grouped validation seeds. If PolyDistill only beats weak PPO but not Direct, state that clearly.

## Baseline sanity
In parallel, reproduce or explain the gap to the public LeRobot model card. Run official-style eval if time permits:
`eval.n_episodes=500`, `eval.batch_size=50`, seed `100000`, temp/sampler exactly documented. Do not block v2 on this, but report deviations.

## Acceptance criteria for a meaningful result
A variant is worth scaling only if at least one is true on held-out validation or paired eval:
- Held-out grouped pass@1 improves over Direct by >= 0.10 without reducing coverage@16.
- Standard paired 100 pass@1 improves over Direct/PPO by >= 0.03.
- Avg max overlap improves by >= 0.03 with pass@1 not worse by more than 0.02.
- Stress pass@1 or avg overlap improves over Direct/PPO without standard collapse.

If no variant passes, produce a mechanistic failure report: overfit train starts, code collapse, KL too strong/weak, top-m label noise, insufficient candidate diversity, baseline mismatch, or invalid selection target.

## Tests / validation
Run focused tests before and after changes:
- PolyDistill tests
- artifact validator
- policy/VQ-BeT PPO tests
- state restore tests
- eval helpers
- no fake grouped pass@k

Minimum command:
`pytest -q tests/test_polydistill.py tests/test_artifact_validator.py tests/test_vqbet_code_ppo.py tests/test_pusht_state_restore.py tests/test_eval_helpers.py`

All generated artifacts must validate. Do not report unvalidated runs.

## Final report requirements
`reports/polyppo_polydistill_v2_results.md` must contain:
- Exact datasets collected: starts x attempts, train/val/test seed manifests.
- Methods and hyperparameters.
- Baselines and checkpoints.
- Held-out grouped pass@k table.
- Standard paired 100/500 table.
- Stress table.
- CIs where available.
- GPU utilization and wall time.
- Best result and whether it meets acceptance criteria.
- Honest interpretation and next step.

## Final response
Report only:
1. PR/branch push status.
2. Tests passed/failed.
3. Datasets collected.
4. Best PolyDistill v2 variant.
5. Direct/PPO/PolyDistill v1/v2 grouped pass@k.
6. Standard paired pass@1 and avg overlap.
7. Stress result.
8. Whether acceptance criteria passed.
9. Exact blocker and resume command if not.
