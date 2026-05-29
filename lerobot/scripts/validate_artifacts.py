#!/usr/bin/env python

import argparse
import json
import sys

from lerobot.common.artifacts.validator import validate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment artifact directories.")
    parser.add_argument("paths", nargs="+", help="Artifact directories to validate.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "benchmark", "final"],
        default="benchmark",
        help="Validation strictness profile.",
    )
    args = parser.parse_args()

    results = [validate_artifact(path, profile=args.profile) for path in args.paths]
    print(
        json.dumps(
            [
                {
                    "path": str(result.path),
                    "artifact_kind": result.artifact_kind,
                    "ok": result.ok,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
                for result in results
            ],
            indent=2,
        )
    )
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
