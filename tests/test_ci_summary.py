"""Required installation jobs must not produce a successful summary on failure."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_installation_results.py"
)
REQUIRED = (
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


def run_checker(payload):
    env = dict(os.environ, NEEDS_RESULTS=payload)
    return subprocess.run(
        [sys.executable, str(CHECKER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def successful_jobs():
    return {name: {"result": "success"} for name in REQUIRED}


def test_all_required_jobs_succeed():
    result = run_checker(json.dumps(successful_jobs()))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("job", REQUIRED)
@pytest.mark.parametrize("state", ["failure", "cancelled", "skipped"])
def test_non_successful_required_job_fails_summary(job, state):
    needs = successful_jobs()
    needs[job]["result"] = state
    result = run_checker(json.dumps(needs))
    assert result.returncode == 1
    assert f"{job}: {state}" in result.stdout


def test_missing_required_job_fails_summary():
    needs = successful_jobs()
    del needs["pip-dev"]
    result = run_checker(json.dumps(needs))
    assert result.returncode == 1
    assert "pip-dev: missing" in result.stdout


@pytest.mark.parametrize("payload", ["not JSON", "[]", "null", "{}"])
def test_invalid_or_empty_results_fail_summary(payload):
    result = run_checker(payload)
    assert result.returncode == 1
