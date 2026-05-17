# PolyPPO Novelty Discovery Blocker

Status: blocked before experiment execution.

## Objective

Review `reports/experiments/polyppo_novelty_discovery_goal.md` and autonomously run the novelty-discovery experiment loop until acceptance criteria pass or report the exact blocker and resume command.

## Audit

- `reports/experiments/polyppo_novelty_discovery_goal.md` is absent from the active clean checkout at `/tmp/bid_lerobot_polyppo_review`.
- The same path is absent from the original `/pub7/neel2/bid_lerobot` checkout.
- The file is not tracked in either checkout.
- Nearby reports cover the previous PolyPPO execution/review goals, not a novelty-discovery loop with explicit acceptance criteria.

## Blocker

The named goal file is missing, so the acceptance criteria, required experiment loop, novelty hypotheses, gates, commands, and deliverables cannot be audited or executed safely. Running experiments from inferred intent would risk producing evidence for the wrong objective.

## Resume Command

Restore or create the goal file first, then resume from the clean PR checkout:

```bash
cd /tmp/bid_lerobot_polyppo_review
test -f reports/experiments/polyppo_novelty_discovery_goal.md
sed -n '1,320p' reports/experiments/polyppo_novelty_discovery_goal.md
```

After the file exists and its acceptance criteria are known, continue by mapping every explicit requirement in that file to an artifact checklist before launching GPU jobs.
