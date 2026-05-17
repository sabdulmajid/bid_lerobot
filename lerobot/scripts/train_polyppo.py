#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.trainer import train_polyppo_one_update


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPO/PolyPPO training for VQ-BeT RVQ code actions.")
    parser.add_argument("--config", required=True, help="Path to a PolyPPO training YAML config.")
    args = parser.parse_args()
    payload = train_polyppo_one_update(args.config)
    checks = payload["checks"]
    print(f"wrote train artifact: {payload['output_path']}")
    for key, value in checks.items():
        print(f"{key}: {value}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
