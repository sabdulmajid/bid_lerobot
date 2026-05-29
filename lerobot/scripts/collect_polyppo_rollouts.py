#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.rollout import collect_polyppo_rollouts


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect set-attempt rollout artifacts for PolyPPO.")
    parser.add_argument("--config", required=True, help="Path to a PolyPPO rollout YAML config.")
    args = parser.parse_args()
    payload = collect_polyppo_rollouts(args.config)
    print(f"wrote rollout artifact: {payload['output_path']}")
    print(f"validation_status: {payload.get('validation_status')}")
    return 0 if payload.get("validation_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
