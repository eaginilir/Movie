"""
Validate task 1 prediction files against data/test.txt.

Checks:
- every test (user, item) pair is present exactly once
- no extra pair appears
- scores are in [10, 100]
- the parser can read every grouped count
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from common import RESULTS_DIR, load_ratings, validate_result_file


def default_result_files():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "*_result.txt")))


def run(files=None):
    test_pairs = load_ratings("test.txt", has_score=False)
    files = files or default_result_files()
    if not files:
        print("No result files found.")
        return 1

    all_ok = True
    for path in files:
        try:
            report = validate_result_file(path, test_pairs=test_pairs)
            all_ok = all_ok and bool(report["ok"])
            status = "OK" if report["ok"] else "FAIL"
            print(f"[{status}] {path}")
            print(
                "  pairs: {actual_pairs}/{expected_pairs}, "
                "missing: {missing_pairs}, extra: {extra_pairs}, "
                "duplicates: {duplicates}, range: {score_min}-{score_max}, "
                "order_matches: {order_matches}".format(**report)
            )
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] {path}")
            print(f"  {exc}")

    return 0 if all_ok else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Validate task 1 result files.")
    parser.add_argument("files", nargs="*", help="Result files to validate. Defaults to results/*_result.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args.files))
