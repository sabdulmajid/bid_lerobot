#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.distill_v2 import train_polydistill_v2


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PolyDistill v2 weighted code CE objectives.")
    parser.add_argument("--config", required=True, help="Path to a PolyDistill v2 training YAML config.")
    args = parser.parse_args()
    payload = train_polydistill_v2(args.config)
    print(f"wrote PolyDistill v2 train artifact: {payload['output_path']}")
    for key, value in payload["checks"].items():
        print(f"{key}: {value}")
    return 0 if all(payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

