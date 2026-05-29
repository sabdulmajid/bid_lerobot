# PolyPPO Next Goal: Turn the Current Signal Into a Novel Result

You are in `sabdulmajid/bid_lerobot` on PR #1.

Mission: stop treating the current outcome as failure. Use it as the discovery signal. Naive code-diversity PolyPPO lost 500-episode pass@1, but improved grouped same-start pass@k/coverage and avg max overlap versus PPO. The next goal is to convert that into a stronger method and result.

Core hypothesis to test:

> Naive diversity rewards can push VQ-BeT off-manifold and hurt pass@1, but quality-gated / return-conditioned diversity can preserve the pass@k coverage gain while recovering pass@1 and robustness.

Do not run huge blind sweeps. Run autonomous, evidence-driven experiments. Use both GPUs.

## 0. First: verify/fix current PR review issues

Before new experiments, inspect PR #1 and fix any remaining live review comments. Do not trust stale local claims.

At minimum verify/fix:
- no random noise is injected into direct single-step eval when `noise_level=0.0`.
- queued VQ-BeT actions do not require latent metadata unless `return_latent=True`.
- training aborts if collect-if-missing rollout validation returns rejected.
- all artifact validators still pass.
- all focused tests still pass.

Run:

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q \
  tests/test_pusht_state_restore.py \
  tests/test_vqbet_code_ppo.py \
  tests/test_polyppo_scaffold.py \
  tests/test_artifact_validator.py \
  tests/test_run_registry.py \
  tests/test_polyppo_metrics.py \
  tests/test_polyppo_rollout_schema.py \
  tests/test_polyppo_trainer_smoke.py \
  tests/test_polyppo_scripts.py
```

## 1. Interpret current evidence correctly

Current evidence:
- Direct VQ-BeT 500 pass@1: about 0.504.
- PPO no-diversity 500 pass@1: about 0.398.
- naive PolyPPO code-div 500 pass@1: about 0.392.
- grouped same-start pass@k improved for PolyPPO vs PPO:
  - PPO pass@1/2/4/8/16 = 0.375/0.625/0.750/0.875/0.875
  - PolyPPO pass@1/2/4/8/16 = 0.500/0.625/0.875/0.875/1.000
  - coverage@16 7/8 -> 8/8
  - avg max overlap 0.7954 -> 0.8788

Do not claim pass@1 improvement. The new research angle is coverage/pass@k improvement and how to recover pass@1.

## 2. Required new methods

Implement and test these variants.

### A. Return-only set advantage
Purpose: separate "set-relative PPO" from "diversity reward".

Score:
```text
score_i = return_i
A_i = normalize_within_set(score_i)
```

### B. Quality-gated code diversity
Reward code diversity only for attempts that are good within the set.

Score:
```text
good_i = 1[return_i >= quantile(return_set, tau)] or 1[return_i >= set_mean]
score_i = return_i + lambda_div * good_i * code_div_i
```

Sweep:
- tau: 0.5, 0.75
- lambda_div: 0.01, 0.03, 0.1

### C. Quality-gated action/trajectory diversity
Code diversity may not equal behavioral diversity. Add action-space and endpoint/trajectory diversity.

Score:
```text
score_i = return_i + lambda_action * good_i * action_div_i + lambda_endpoint * good_i * endpoint_div_i
```

Sweep small:
- lambda_action: 0.01, 0.03, 0.1
- endpoint if cheap; otherwise action diversity first.

### D. Bad-diversity penalty
Penalize diversity among bad attempts to avoid off-manifold exploration.

Score:
```text
bad_i = 1[return_i < set_mean]
score_i = return_i + lambda_good * good_i * div_i - lambda_bad * bad_i * div_i
```

Try lambda_good/lambda_bad:
- 0.03/0.03
- 0.1/0.03
- 0.1/0.1

### E. KL-to-base and success-BC regularization
Naive PPO degraded direct VQ-BeT. Add regularizers.

Required variants:
- PPO + KL-to-base.
- quality-gated PolyPPO + KL-to-base.
- quality-gated PolyPPO + success-only BC/logprob bonus on top attempts.

Do not include fake KL: if base logprobs are missing, fix them or disable that cell.

### F. Best-of-k distillation / PolyDistill
This is the high-upside novelty path.

After grouped pass@k finds successful/high-overlap attempts from the same prefix, distill the best attempt's RVQ code sequence back into the policy.

Implement lightweight version:
- collect grouped sets, k=8 or 16.
- select top attempt by return/max overlap per set.
- add supervised CE loss on selected code ids.
- combine with KL-to-base and optional PPO loss.

Compare:
- Direct VQ-BeT
- PPO no-div
- naive PolyPPO code-div
- Quality-gated PolyPPO
- PolyDistill top-of-set

Target claim if supported:
> pass@k reveals diverse successful continuations; top-of-set distillation recovers some of that coverage into pass@1.

## 3. Baselines that must be added

Current pass@k comparison only against PPO is insufficient. Add grouped same-start baselines:

- Direct pretrained VQ-BeT, stochastic samples, same starts, k=1/2/4/8/16.
- PPO no-diversity.
- naive PolyPPO code-div.
- quality-gated PolyPPO.
- PolyDistill if implemented.

Use same prefix states and seed manifest across methods.

## 4. Baseline reproduction investigation

The direct VQ-BeT baseline around 0.50 is below the public model-card number. Investigate, but do not let this block paired comparisons.

Run/record:
- exact checkpoint path/hash.
- direct sampler temperature.
- eval seed range.
- batch size.
- environment version.
- action horizon / AH_test.
- whether direct/noise path is disabled when noise=0.

Run 100-episode official-style reproduction, then 500 if time allows. If still low, explain deviation in report.

## 5. GPU execution plan

Use both RTX PRO 6000 GPUs.

GPU 0:
- grouped pass@k evals.
- baseline reproduction.
- stress evals.

GPU 1:
- 32x8 and 64x8 rollout collection.
- PPO/PolyPPO/PolyDistill training.

Run successive halving:
1. Tiny smoke: 4x3, 10 eval episodes.
2. Real triage: 32x8, 100 paired eval episodes.
3. Candidate confirmation: 64x8, 200 paired eval episodes.
4. Winner scale: 500 paired eval episodes.

Only promote configs that beat PPO no-div or direct VQ-BeT on a meaningful metric.

## 6. Stress conditions

Evaluate at least:
- standard PushT.
- observation noise.
- action noise.
- randomized/held-out starts if implemented.

Do not present stress as leaderboard. Present it as robustness analysis.

## 7. Metrics

Required:
- pass@1.
- grouped pass@2/4/8/16.
- coverage@k: number of same-start groups solved by at least one attempt.
- avg max overlap.
- success CI95 bootstrap.
- action diversity.
- code diversity.
- endpoint diversity if available.
- KL-to-base.
- entropy.
- wall time/GPU metadata.

Also report negative results. Do not hide pass@1 degradation.

## 8. Acceptance criteria

Complete this goal only when:

1. Remaining PR review comments are fixed or explicitly shown stale/resolved.
2. Focused tests pass.
3. Direct VQ-BeT grouped pass@k baseline exists.
4. At least one quality-gated PolyPPO or PolyDistill run completes beyond one update.
5. At least one 100-episode paired eval exists for Direct, PPO no-div, naive PolyPPO, and the new best method.
6. Report clearly states whether novelty was found.

A strong result is one of:
- Quality-gated PolyPPO improves pass@k/coverage over direct and PPO while limiting pass@1 regression.
- PolyDistill converts pass@k gains into pass@1 improvement.
- Robustness-augmented/quality-gated PolyPPO beats PPO under observation noise.

If none happen, write a negative-result report explaining why naive diversity helps coverage but harms pass@1/robustness.

## 9. Reports

Update/create:
- `reports/polyppo_novelty_search.md`
- `reports/polyppo_novelty_summary.json`

Report structure:
1. Current evidence and interpretation.
2. Why naive code diversity failed pass@1.
3. Grouped direct/PPO/PolyPPO pass@k table.
4. New methods: return-only, quality-gated, bad-div penalty, KL/BC, PolyDistill.
5. Standard/stress results.
6. Best supported claim.
7. Limitations.
8. Exact commands and resume command.

## 10. Final response

Return only:
- PR review status.
- tests passed/failed.
- experiments run.
- direct/PPO/naive PolyPPO/new-method metrics.
- best novelty signal.
- whether it is positive, mixed, or negative.
- exact blocker if any.
- next resume command.
