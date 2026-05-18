#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.distill import collect_polydistill_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect top-of-set PolyDistill datasets.")
    parser.add_argument("--config", required=True, help="Path to a PolyDistill collection YAML config.")
    args = parser.parse_args()
    payload = collect_polydistill_dataset(args.config)
    print(f"wrote PolyDistill dataset: {payload['output_path']}")
    print(f"validation_status: {payload.get('validation_status')}")
    return 0 if payload.get("validation_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
