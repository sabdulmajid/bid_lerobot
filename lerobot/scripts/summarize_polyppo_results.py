#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from lerobot.common.polyppo.storage import load_rollout_artifact
from lerobot.common.polyppo.utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create PolyPPO report artifacts from rollout/train outputs.")
    parser.add_argument("--output-dir", required=True, help="PolyPPO run directory to summarize.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = {"output_dir": str(output_dir), "rollouts": [], "training": [], "evaluations": []}
    for rollout_path in output_dir.rglob("ppo_rollout.json"):
        payload, _ = load_rollout_artifact(rollout_path)
        summary["rollouts"].append(
            {
                "path": str(rollout_path.parent),
                "run_id": payload.get("run_id"),
                "n_sets": payload.get("n_sets"),
                "n_attempts": payload.get("n_attempts"),
                "aggregated": payload.get("aggregated"),
                "validation_status": payload.get("validation_status"),
            }
        )
    for train_path in output_dir.rglob("polyppo_train_info.json"):
        payload = json.loads(train_path.read_text())
        summary["training"].append(
            {
                "path": str(train_path.parent),
                "run_id": payload.get("run_id"),
                "checks": payload.get("checks"),
                "loss": payload.get("loss"),
                "checkpoint_path": payload.get("checkpoint_path"),
            }
        )
    for eval_path in output_dir.rglob("eval_info.json"):
        payload = json.loads(eval_path.read_text())
        summary["evaluations"].append(
            {
                "path": str(eval_path.parent),
                "sampler": payload.get("run_metadata", {}).get("sampler"),
                "n_episodes": payload.get("run_metadata", {}).get("n_episodes"),
                "checkpoint": payload.get("run_metadata", {}).get("checkpoint"),
                "aggregated": payload.get("aggregated"),
            }
        )
    for eval_summary_path in output_dir.rglob("eval_polyppo_summary.json"):
        payload = json.loads(eval_summary_path.read_text())
        summary.setdefault("eval_summaries", []).append(
            {
                "path": str(eval_summary_path.parent),
                "rows": payload.get("rows", []),
            }
        )
    write_json(output_dir / "polyppo_pusht_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
