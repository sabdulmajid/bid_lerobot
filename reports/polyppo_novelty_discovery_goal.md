# PolyPPO Novelty Discovery Goal

You are in `sabdulmajid/bid_lerobot`. The objective is no longer just “make PolyPPO run.” The objective is to **discover whether a novel PolyPPO-style post-training method can improve VQ-BeT PushT pass@k, coverage, robustness, or pass@1**.

Do not stop because the first code-diversity one-update config failed under observation noise. Treat that as a scientific signal: the naive diversity objective may be rewarding brittle/off-manifold behavior. Push forward with controlled, aggressive experimentation.

## Current factual state

- PR #1 added PolyPPO foundation and evidence ladder.
- VQ-BeT PPO action identity is RVQ code-id sequence only.
- Current evidence: direct VQ-BeT baseline 100 episodes pass@1 0.50 / avg max overlap 0.7667; rollout smoke 4x3; one-update PPO and PolyPPO pass gates; stress triage shows PolyPPO code-div +0.05 standard but -0.20 under observation noise.
- Focused suite previously passed: 59 passed, 2 warnings.
- Existing constrained BID artifacts are invalid because strong/reference hashes are identical.

## Mission

Produce real novelty-seeking evidence:

> Can set-relative PolyPPO over VQ-BeT RVQ code actions improve test-time coverage/pass@k or stress robustness, without collapsing pass@1, compared with PPO without diversity and pretrained VQ-BeT?

## Mandatory operating rule

Do not “fall back” to only diagnosing. Diagnose by running experiments. Keep training/eval GPUs busy.

## Subagents / workstreams

If subagents exist, launch them. Otherwise emulate as parallel workstreams in `outputs/polyppo/status.md`.

A. Review + correctness
- Pull latest PR branch.
- Verify all current PR comments are truly resolved.
- If GitHub still shows P1/P2 comments, fix them first.
- Run focused tests.

B. Baseline + pass@k
- Reproduce official-style VQ-BeT PushT eval.
- Implement real grouped pass@k: repeated attempts from the same start state, not consecutive unrelated seeds.
- Run temperature/sampler baseline sweep.

C. PolyPPO search
- Run 32x8 and 64x8 rollout batches.
- Train PPO and PolyPPO variants.
- Use successive halving: triage 50 episodes, promote to 100, promote winners to 500.

D. Stress + analysis
- Standard PushT, observation noise, action noise, held-out/randomized starts, reduced reactivity/chunk length.
- Summarize results and failure modes.

GPU plan:
- GPU 0: eval, grouped pass@k, stress sweeps.
- GPU 1: training and rollout collection.
- Run concurrent jobs when possible. Log GPU id, command, git SHA, dirty flag, wall time, checkpoint hash.

## Why the observation-noise failure may be happening

Investigate these hypotheses while running experiments:

1. One-update/20-episode stress is too noisy; confidence intervals are wide.
2. Code diversity may push into latent codes that are fine in clean observations but brittle under corrupted observations.
3. Diversity reward may be unnormalized or too large relative to return.
4. No robustness augmentation was used during training, so observation-noise eval is OOD.
5. PPO action is code ids only; deterministic continuous decoder/offset behavior may still be brittle.
6. Baseline eval may not match official VQ-BeT settings; current 0.50 is below the model-card reference.
7. pass@k is not yet valid unless repeated attempts use the same start state.

Do not use these as excuses. Use them to design experiments.

## Phase 0: latest PR review check

1. Fetch latest branch and inspect PR comments.
2. Fix any unresolved P1/P2 comments.
3. Confirm:
   - value head checkpoint migration works for pretrained VQ-BeT.
   - config.json policy type is preserved.
   - grouped pass@k rejects invalid grouping unless same start state is repeated.
   - valid-mask PPO loss is applied.
   - GPU id selection is honored.
   - observation-noise stress actually modifies observations.
   - KL-to-base is either real or disabled/labelled honestly.
4. Run focused tests.

## Phase 1: baseline reproduction and sampler sweep

The current baseline is lower than expected. Run official-style baseline before interpreting improvement.

Run:
- 100 episode VQ-BeT eval with official-ish settings.
- 500 episode eval if feasible.
- Temperature sweep: 0.05, 0.1, 0.3, 0.5.
- Sampler modes already supported: direct, stochastic/code sampling if available.

Acceptance:
- Document exact checkpoint hash and command.
- Explain why result differs from official model-card numbers if it still does.
- Do not block PolyPPO search on perfect reproduction; run in parallel.

## Phase 2: real grouped pass@k evaluator

Implement or fix evaluator so pass@k uses repeated attempts from the **same initial state**.

For each start state:
- restore/replay same initial state;
- run k stochastic attempts;
- compute pass@1, pass@2, pass@4, pass@8, pass@16;
- log attempt-level successes and max overlap.

Reject pass@k if starts are not grouped correctly.

## Phase 3: rollout collection at real batch size

Collect:
- smoke: 8 prefixes x 4 attempts;
- main: 32 prefixes x 8 attempts;
- stretch: 64 prefixes x 8 attempts.

Store per attempt:
- set_id, attempt_id, prefix state hash, old logprob, value, entropy, code ids, action summary, rewards, done, success, max overlap, final return, diversity metrics.

## Phase 4: novelty-seeking PolyPPO sweep

Run these methods:

1. pretrained VQ-BeT
2. PPO no diversity
3. PPO + KL-to-base
4. PPO + BC/behavior regularization if cheap
5. PolyPPO return-only set advantage
6. PolyPPO code diversity
7. PolyPPO action diversity
8. PolyPPO endpoint/trajectory diversity if cheap
9. quality-gated diversity: diversity bonus only among attempts above a return/overlap threshold
10. robust PolyPPO: train/evaluate with stress mixture or observation augmentation

Hyperparameters:
- lambda_div: 0.0, 0.01, 0.03, 0.1, 0.3
- lambda_kl_base: 0.0, 0.01, 0.05
- entropy_coef: 0.0, 0.003, 0.01
- attempts_per_prefix: 4, 8
- n_prefixes: 32, 64
- learning rate: include at least 3 values around current default

Use successive halving:
- Stage 1: 50 eval episodes per config.
- Stage 2: top 25% get 100 paired episodes.
- Stage 3: top 2-3 get 500 paired episodes.

Do not scale only the failed one-update code-div checkpoint. Scale the search.

## Phase 5: stress and robustness discovery

Evaluate top candidates on:
- standard PushT;
- observation noise;
- action noise;
- randomized/held-out starts;
- reduced reactivity or longer action chunk execution if implemented.

Primary success metrics:
- pass@1;
- pass@4/pass@8 grouped correctly;
- avg max overlap;
- coverage: number of unique start states solved over k attempts;
- robustness delta under stress;
- diversity without pass@1 collapse;
- wall-clock/sample cost.

Novelty target:
- If PolyPPO improves pass@k/coverage while matching pass@1, that is meaningful.
- If robust/quality-gated PolyPPO improves observation-noise robustness versus PPO, that is stronger.
- If PolyPPO loses everywhere, produce a rigorous negative result with diagnostics.

## Phase 6: combine training + inference-time diversity

If a PolyPPO checkpoint shows any useful signal, evaluate:
- PolyPPO checkpoint + direct sampler;
- PolyPPO checkpoint + BID if valid checkpoint pair exists;
- PolyPPO checkpoint + PolySelect / multi-sample selection;
- pretrained VQ-BeT + same inference method.

This tests whether training improves the candidate pool used by inference-time selection.

## Commands to run/adapt

Run tests:

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q tests/test_policy_factory.py tests/test_eval_helpers.py tests/test_polyppo_scripts.py tests/test_sampler_return_contracts.py tests/test_vqbet_code_ppo.py tests/test_polyppo_scaffold.py tests/test_artifact_validator.py tests/test_polyppo_trainer_smoke.py tests/test_polyppo_utils.py tests/test_polyppo_rollout_schema.py tests/test_polyppo_metrics.py tests/test_pusht_state_restore.py tests/test_run_registry.py
```

Then run/adapt:

```bash
nvidia-smi
python -m lerobot.scripts.eval --policy.path=<STRONG_VQBET_CHECKPOINT> --output_dir=outputs/eval/vqbet_baseline_500 --env.type=pusht --seed=100000 --eval.n_episodes=500 --eval.batch_size=50 --device=cuda --use_amp=false
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.collect_polyppo_rollouts --config configs/polyppo/pusht_rollout_32x8.yaml
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_small_sweep.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval_polyppo_checkpoints --config configs/polyppo/pusht_grouped_passk_stress.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.summarize_polyppo_results --output reports/polyppo_novelty_summary.json
```

If script names differ, use the implemented equivalent and record exact commands.

## Report requirements

Create/update:
- `reports/polyppo_novelty_search.md`
- `reports/polyppo_novelty_summary.json`

Include:
- exact hypothesis tested;
- hardware/GPU usage;
- checkpoint hashes;
- baseline reproduction;
- grouped pass@k method;
- 32x8/64x8 rollout stats;
- PPO vs PolyPPO sweep table;
- top configs promoted and why;
- stress results;
- failures and diagnostics;
- confidence intervals;
- exact commands;
- next experiment.

## Acceptance criteria

Minimum completion:
- all focused tests pass;
- PR review issues verified/fixed;
- official-style baseline rerun or explicitly running in parallel;
- 32x8 rollout batch collected;
- at least one multi-update PPO and one multi-update PolyPPO run completed;
- grouped pass@k evaluator works or invalid pass@k is rejected;
- at least 100 paired eval episodes for top PPO vs top PolyPPO;
- report contains numeric results.

Strong completion:
- 500 paired eval episodes for pretrained VQ-BeT, PPO no-diversity, best PolyPPO;
- standard + at least two stress settings;
- best PolyPPO improves pass@k/coverage or robustness at matched pass@1, or produces a rigorous negative result explaining why not.

## Final response format

Report only:
1. PR review status.
2. Baseline reproduction result.
3. Rollout batch size collected.
4. PPO configs run.
5. PolyPPO configs run.
6. Best result and confidence interval.
7. Grouped pass@k result.
8. Stress result.
9. Whether novelty signal exists.
10. Files changed.
11. Commands run.
12. Exact blocker/resume command if incomplete.
