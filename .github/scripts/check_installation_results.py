"""Fail the installation summary unless every required job succeeded."""

import json
import os

REQUIRED_JOBS = (
    "pip-core",
    "pip-brainda",
    "pip-brainflow",
    "pip-full",
    "pip-dev",
    "pip-dev-requirements",
    "conda-install",
    "uv-install",
    "build-wheel",
)


def main():
    try:
        needs = json.loads(os.environ.get("NEEDS_RESULTS", ""))
    except json.JSONDecodeError:
        print("Invalid or missing installation results")
        return 1
    if not isinstance(needs, dict):
        print("Installation results must be a JSON object")
        return 1

    failed = []
    for job in REQUIRED_JOBS:
        details = needs.get(job)
        result = (
            details.get("result", "missing") if isinstance(details, dict) else "missing"
        )
        print(f"{job}: {result}")
        if result != "success":
            failed.append(job)
    if failed:
        print("Required installation jobs did not succeed: " + ", ".join(failed))
        return 1
    print("All required installation jobs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
