# PolyPPO PushT Results

Status: real Phase 1-3 evidence and triage stress evidence exist, but this is not yet publishable evidence. The missing piece is still a real multi-update PPO/PolyPPO sweep at larger rollout scale.

Headline claim: not filled. Current runs are smoke and triage scale only.

## Scope

- PPO action identity: RVQ code-id sequence.
- Continuous offsets are deterministic VQ-BeT decoder/head outputs conditioned on observations and sampled RVQ code IDs.
- This is code-policy PPO, not full continuous-action PPO.
- Hardware: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB each.
- Main checkpoint: `lerobot/vqbet_pusht`, snapshot `15c2d0af889c401c7e5db7a07499d3b498afc276`, `model.safetensors` SHA256 `f60b22049b275c026159fd4fcc018721ffbec47119ecfab98dc84814eb4e0d30`.

## Evidence

| Artifact | Episodes / Sets | Result | Validation |
|---|---:|---|---|
| Pretrained VQ-BeT direct, temp 0.1 | 100 episodes | pass@1 0.50, avg max overlap 0.7667, avg steps 232.2 | passed benchmark |
| Real set-attempt rollout | 4 sets x 3 attempts | pass@1/2/4/8 all 0.0, avg code diversity 1.0521, avg action diversity 1.6109 | passed smoke |
| PPO no diversity one update | 10 post-update eval episodes | pass@1 0.50, avg max overlap 0.8478 | passed smoke |
| PolyPPO code diversity one update | 10 post-update eval episodes | pass@1 0.50, avg max overlap 0.8478 | passed smoke |
| PPO no diversity stress standard | 20 episodes | pass@1 0.55, avg max overlap 0.7886 | passed benchmark |
| PPO no diversity stress action noise | 20 episodes | pass@1 0.70, avg max overlap 0.8881 | passed benchmark |
| PolyPPO code diversity stress standard | 20 episodes | pass@1 0.60, avg max overlap 0.8279 | passed benchmark |
| PolyPPO code diversity stress action noise | 20 episodes | pass@1 0.65, avg max overlap 0.8718 | passed benchmark |

The stress numbers are not statistically strong. The 95% CIs are wide at 20 episodes. They are useful as a wiring and triage signal only.

## Correctness Gates

- Official-temperature VQ-BeT eval now runs through the artifact validator.
- VQ-BeT sampled code log-prob tests pass.
- Recomputed old log-probs match exactly when replayed with the same microbatch path.
- PPO ratio before update is exactly 1.0 for PPO and PolyPPO one-update runs.
- Value-head overfit, PushT state restore/prefix replay, set-normalized advantages, and `lambda_div=0` equivalence are covered by focused tests.
- Real rollout artifacts store set ids, attempt ids, prefix hashes, code ids, old log-probs, values, entropy, rewards, done/success, max overlap, returns, action/code diversity, checkpoint hashes, command, git metadata, GPU metadata, and seed manifest.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.eval -p lerobot/vqbet_pusht \
  --out-dir outputs/eval/vqbet_pusht_direct_100_temp01 \
  --sampler direct --temperature 0.1 \
  eval.n_episodes=100 eval.batch_size=50 device=cuda use_amp=false

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.collect_polyppo_rollouts \
  --config configs/polyppo/pusht_rollout_smoke.yaml

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.train_polyppo \
  --config configs/polyppo/pusht_one_update.yaml

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.train_polyppo \
  --config configs/polyppo/pusht_one_update_polyppo_code.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.eval_polyppo_checkpoints \
  --config configs/polyppo/pusht_stress_eval.yaml

PYTHONPATH=/tmp/gymnasium_vendor:/tmp/termcolor_pkg:$PYTHONPATH \
python -m lerobot.scripts.summarize_polyppo_results --output-dir outputs/polyppo
```

## Interpretation

- PolyPPO code-diversity one update did not clearly beat PPO no-diversity at matched triage compute. It is +0.05 pass@1 on 20-episode standard stress and -0.05 pass@1 under action noise, both inside wide uncertainty.
- The action-noise improvement for both methods is likely seed/noise interaction at small N, not a reliable robustness claim.
- The rollout smoke uses only 4 prefix sets x 3 attempts and short continuation horizon; it proves infrastructure, not learning quality.
- The old log-prob mismatch blocker was real and fixed by using the same one-sample recompute path as online collection for the hard ratio gate.

## Missing For Publishable Evidence

- Full multi-update PPO/PolyPPO training loop with fresh rollout collection or a clearly documented on-policy batch schedule.
- Real small sweep over at least PPO no-diversity, PPO+KL/BC regularization, PolyPPO return-only, PolyPPO code diversity, and PolyPPO action diversity.
- Larger rollout config: at least 32 prefix sets x 8 attempts for candidate configs.
- 100 episode paired eval for promising configs and 200-500 paired episodes for winners.
- Stress variants beyond action noise: observation noise and held-out/randomized starts.
- BID and PolySelect inference baselines; they remain baselines and should not block PolyPPO training.

## Next Run

Implement the small sweep as real repeated training, then run 50 paired eval episodes for:

1. PPO no diversity.
2. PPO + KL-to-base.
3. PolyPPO return-only.
4. PolyPPO code diversity.
5. PolyPPO action diversity.

Kill scaling if PolyPPO does not beat PPO no-diversity or pretrained direct on pass@k, coverage/max overlap, robustness, or pass@1 at matched compute.
