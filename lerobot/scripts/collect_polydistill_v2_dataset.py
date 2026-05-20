#!/usr/bin/env python

import argparse

from lerobot.common.polyppo.distill_v2 import collect_polydistill_v2_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PolyDistill v2 coverage-compression datasets.")
    parser.add_argument("--config", required=True, help="Path to a PolyDistill v2 collection YAML config.")
    args = parser.parse_args()
    payload = collect_polydistill_v2_dataset(args.config)
    print(f"wrote PolyDistill v2 dataset: {payload['output_path']}")
    print(f"validation_status: {payload.get('validation_status')}")
    return 0 if payload.get("validation_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

