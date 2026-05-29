#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.distill import train_polydistill


def main() -> int:
    parser = argparse.ArgumentParser(description="Train VQ-BeT code logits from top-of-set PolyDistill data.")
    parser.add_argument("--config", required=True, help="Path to a PolyDistill training YAML config.")
    args = parser.parse_args()
    payload = train_polydistill(args.config)
    print(f"wrote PolyDistill train artifact: {payload['output_path']}")
    for key, value in payload["checks"].items():
        print(f"{key}: {value}")
    return 0 if all(payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
