"""Fetch the next AbleFAST coupon and refresh its two model draw picks."""

import csv
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


BASE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_FIXTURES = 49


def run_step(script):
    print("\n---", script, "---")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, script)],
        cwd=BASE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise SystemExit(f"{script} failed with exit code {result.returncode}.")


def fixture_count():
    path = os.path.join(BASE, "pool_fixtures.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if row.get("HomeTeam") and row.get("AwayTeam"))


def main():
    run_step("fetch_pool_fixtures.py")
    run_step("make_pool_file.py")
    count = fixture_count()
    if count != EXPECTED_FIXTURES:
        raise SystemExit(
            f"Only {count} of {EXPECTED_FIXTURES} fixtures matched the model data. "
            "Fix the unmatched team names before rating the coupon."
        )
    run_step("pool_draws.py")


if __name__ == "__main__":
    main()
