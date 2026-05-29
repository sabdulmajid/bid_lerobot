# PolyPPO Novelty Search

Status: minimum and strong PushT evidence criteria completed on commit `972ace1c121556307960f6e1f8db85b07ac82b15`.

## Hypothesis

Set-relative PolyPPO over VQ-BeT RVQ code IDs can improve test-time candidate coverage/pass@k by shifting the sampled code pool, but naive code diversity may not improve single-attempt pass@1 or stress robustness.

Action definition: PPO/PolyPPO optimizes only the current-token RVQ code-id tuple. Continuous offsets and decoded continuous actions remain deterministic outputs of the VQ-BeT decoder/head conditioned on the sampled/replayed code IDs.

## Correctness Gates

- Focused suite: `63 passed, 2 warnings`.
- PR review state: unresolved P1/P2 issues found and fixed in `972ace1`:
  - value-head migration allowance for pretrained VQ-BeT checkpoints;
  - fake grouped pass@k rejected in flat `eval_policy`;
  - reference Hub checkpoints resolved before metadata hashing;
  - config.json fallback refuses non-VQ-BeT policy types;
  - biased sequential-head entropy proxy disabled and real PushT PPO configs use `entropy_coef: 0.0`.
- Existing gates remain covered by tests/artifacts: sampled code log-prob correctness, old-logprob equality, PPO ratio=1 before update, value-head overfit, PushT state restore determinism, set-normalized advantages, lambda_div=0 behavior, valid-mask PPO loss, GPU id selection, observation-noise application, and strict artifact validation.

Focused command:

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH pytest -q \
  tests/test_policy_factory.py tests/test_eval_helpers.py tests/test_polyppo_scripts.py \
  tests/test_sampler_return_contracts.py tests/test_vqbet_code_ppo.py tests/test_polyppo_scaffold.py \
  tests/test_artifact_validator.py tests/test_polyppo_trainer_smoke.py tests/test_polyppo_utils.py \
  tests/test_polyppo_rollout_schema.py tests/test_polyppo_metrics.py tests/test_pusht_state_restore.py \
  tests/test_run_registry.py
```

## Hardware And Provenance

- GPUs: two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition cards, ~97.9 GiB each.
- GPU policy: polled at least three times before launching each new GPU job after user requested standby.
- Main Git SHA: `972ace1c121556307960f6e1f8db85b07ac82b15`; all reported main artifacts have `git_dirty=false`.
- Pretrained VQ-BeT checkpoint hash: `f60b22049b275c026159fd4fcc018721ffbec47119ecfab98dc84814eb4e0d30`.
- PPO checkpoint hash: `f1c1545678ed3165bb4a744610ad626de11cf310628801aae86b1b16dfe37b1e`.
- PolyPPO code-diversity checkpoint hash: `39459c65f2be9a8708fef7c0ebd5a88655ec7d8a1e85bba03d895216b8b7f121`.

## Baselines

Official-style direct VQ-BeT, 500 episodes, temperature 0.1:

| Seed | pass@1 | Avg max overlap | Success CI95 | Artifact |
| --- | ---: | ---: | --- | --- |
| 100000 | 0.488 | 0.7503 | not recomputed in row artifact | `outputs/eval/vqbet_pusht_novelty_baseline_500_temp01_972ace1` |
| 180000 | 0.504 | 0.7503 | paired seed block | `outputs/eval/vqbet_pusht_novelty_baseline_500_seed180000_temp01_972ace1` |

The direct baseline remains below the advertised model-card-style expectation. This run uses the repository evaluator, direct sampler, no AMP, PushT `eval.n_episodes=500`, and the exact checkpoint hash above. The lower score is therefore treated as the local reproducibility baseline, not as evidence that PolyPPO has improved the official model.

## Rollout Batch

32 prefixes x 8 attempts, GPU 1:

| Metric | Value |
| --- | ---: |
| Attempts | 256 |
| Avg return | 0.5702 |
| Avg max overlap | 0.0696 |
| Avg success | 0.0000 |
| Avg code diversity | 1.2979 |
| Avg action diversity | 4.4521 |
| pass@1/2/4/8 | 0.0 / 0.0 / 0.0 / 0.0 |

Artifact: `outputs/polyppo/pusht_rollout_32x8`, validation passed.

64x8 was not run because the 32x8 batch was sufficient to produce the first publishable signal and the kill/diagnosis decision is now about objective design rather than batch size.

## Multi-Update Training

Both methods trained for `num_updates=3` on the same 32x8 rollout batch; ratio error before update was exactly `0.0`.

| Method | Checkpoint | Final policy loss | Final value loss | Final KL | 100-episode pass@1 | Avg max overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PPO no diversity | `outputs/polyppo/pusht_novelty_ppo_32x8/checkpoint.pt` | -0.000022 | 1.0701 | 0.000002 | 0.390 | 0.7718 |
| PolyPPO code diversity | `outputs/polyppo/pusht_novelty_polyppo_code_32x8/checkpoint.pt` | -0.000073 | 1.1524 | -0.000003 | 0.380 | 0.7742 |

The 100-episode seed manifests match between PPO and PolyPPO.

## 500-Episode Paired Eval

Seed block `180000`, same seed manifest across direct baseline, PPO, and PolyPPO:

| Method | Episodes | pass@1 | Success CI95 | Avg max overlap | Max-overlap CI95 |
| --- | ---: | ---: | --- | ---: | --- |
| Direct VQ-BeT | 500 | 0.504 | not stored by `eval.py` row | 0.7503 | not stored by `eval.py` row |
| PPO no diversity | 500 | 0.398 | [0.3551, 0.4409] | 0.7613 | [0.7306, 0.7920] |
| PolyPPO code diversity | 500 | 0.392 | [0.3492, 0.4348] | 0.7612 | [0.7305, 0.7920] |

Conclusion: this naive code-diversity PolyPPO does not improve pass@1. It is essentially tied with PPO on overlap and slightly worse on pass@1 at matched eval compute.

## Grouped Same-Start Pass@k

Method: `lerobot/scripts/eval_grouped_passk.py` snapshots each PushT start state and restores that exact state for each attempt. It rejects `k > attempts_per_start`. Artifacts record `grouped_same_start=true`, 8 start states, 16 attempts/start, and start-state hashes.

| Method | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | Coverage@16 | Avg max overlap | Max-overlap CI95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PPO no diversity | 0.375 | 0.625 | 0.750 | 0.875 | 0.875 | 7/8 | 0.7954 | [0.7504, 0.8404] |
| PolyPPO code diversity | 0.500 | 0.625 | 0.875 | 0.875 | 1.000 | 8/8 | 0.8788 | [0.8505, 0.9071] |

This is the main positive signal: code-diversity PolyPPO improves grouped coverage and best-overlap on the 8-start grouped benchmark. It also improves pass@4/pass@16 in this small grouped run. The sample is too small to claim robust publishability by itself.

## Stress Results

50 single-attempt episodes per method and variant. Stress pass@k>1 is intentionally not reported here because grouped repeated starts are required.

| Variant | PPO pass@1 | PPO avg max | PolyPPO pass@1 | PolyPPO avg max | Signal |
| --- | ---: | ---: | ---: | ---: | --- |
| Standard | 0.38 | 0.7433 | 0.36 | 0.7721 | PolyPPO improves overlap, not success |
| Action noise 0.03 | 0.40 | 0.7845 | 0.44 | 0.8313 | weak positive PolyPPO signal |
| Observation noise 0.03 | 0.40 | 0.7360 | 0.36 | 0.7770 | overlap up, success down |

The confidence intervals are wide at 50 episodes. Stress results do not justify scaling naive code diversity as a robustness method.

## Decision

Do not scale naive code-diversity PolyPPO further as the headline pass@1 method. It fails the kill criterion against PPO/direct baseline on 500-episode pass@1. The publishable novelty angle, if pursued, should be narrower: PolyPPO as a set-relative training method that improves grouped same-start candidate coverage/pass@k and best overlap, with a clear warning that naive diversity can hurt single-attempt success.

Promote next:

1. Quality-gated diversity: only reward diversity among attempts above a return/overlap threshold.
2. Action/trajectory diversity rather than raw code diversity.
3. Return-only set advantage to isolate the set-normalization contribution.
4. Robust PolyPPO with observation-noise augmentation if robustness is the target.

## Exact Commands

```bash
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval -p lerobot/vqbet_pusht --out-dir outputs/eval/vqbet_pusht_novelty_baseline_500_temp01_972ace1 --sampler direct --temperature 0.1 --max-episodes-rendered 0 eval.n_episodes=500 eval.batch_size=50 device=cuda:0 use_amp=false
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval -p lerobot/vqbet_pusht --out-dir outputs/eval/vqbet_pusht_novelty_baseline_500_seed180000_temp01_972ace1 --sampler direct --temperature 0.1 --max-episodes-rendered 0 eval.n_episodes=500 eval.batch_size=50 seed=180000 device=cuda:1 use_amp=false
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.collect_polyppo_rollouts --config configs/polyppo/pusht_rollout_32x8.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_novelty_ppo_32x8.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.train_polyppo --config configs/polyppo/pusht_novelty_polyppo_code_32x8.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval_grouped_passk --config configs/polyppo/pusht_grouped_passk_novelty.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval_polyppo_checkpoints --config configs/polyppo/pusht_novelty_stress_eval.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval_polyppo_checkpoints --config configs/polyppo/pusht_novelty_ppo_eval_500_seed180000.yaml
PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH python -m lerobot.scripts.eval_polyppo_checkpoints --config configs/polyppo/pusht_novelty_polyppo_code_eval_500_seed180000.yaml
```

