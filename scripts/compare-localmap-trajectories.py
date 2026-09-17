#!/usr/bin/env python3
"""Evaluate a local-map change with Sounding, requiring identical scan coverage.

Usage: python3 scripts/compare-localmap-trajectories.py BASELINE CANDIDATE REFERENCE
Resolution trials may explicitly allow small scan-end timestamp shifts with
--max-timestamp-shift SECONDS; timestamps are never rewritten.
Exit 0 means all four RMSE metrics are non-increasing; 1 means regression;
2 means invalid input or evaluation failure. Sounding must be on PATH.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


METRICS = (
    "ape_trans_rmse",
    "ape_rot_rmse_deg",
    "rpe_trans_rmse",
    "rpe_rot_rmse_deg",
)


def timestamps(path):
    stamps = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [float(field) for field in line.split()]
        if len(fields) != 8 or not all(math.isfinite(value) for value in fields):
            raise ValueError(f"{path}:{number}: expected eight finite TUM values")
        stamp = fields[0]
        if stamps and stamp <= stamps[-1]:
            raise ValueError(f"{path}:{number}: timestamps must increase strictly")
        stamps.append(stamp)
    if len(stamps) < 2:
        raise ValueError(f"{path}: trajectory needs at least two poses")
    return stamps


def evaluate(estimate, reference):
    # Sounding expects ESTIMATED_FILE first, REFERENCE_FILE second.
    completed = subprocess.run(
        ["sounding", "trajectory", str(estimate), str(reference), "--delta", "1", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    for key in METRICS:
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Sounding returned an invalid {key}: {value!r}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, help="Also save the JSON report here")
    parser.add_argument(
        "--max-timestamp-shift", type=float, default=0.0,
        help="Allow bounded per-scan timestamp shifts for resolution trials (default: exact)",
    )
    args = parser.parse_args()
    baseline_stamps = timestamps(args.baseline)
    candidate_stamps = timestamps(args.candidate)
    tolerance = args.max_timestamp_shift
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Timestamp shift limit must be finite and nonnegative")
    if len(baseline_stamps) != len(candidate_stamps):
        raise ValueError(
            "Trajectories must contain the same number of scans; "
            f"baseline has {len(baseline_stamps)} poses, candidate has {len(candidate_stamps)}. "
            "Do not compare a partial replay against the full baseline."
        )
    shifts = [abs(a - b) for a, b in zip(baseline_stamps, candidate_stamps)]
    minimum_interval = min(
        b - a for stamps in (baseline_stamps, candidate_stamps)
        for a, b in zip(stamps, stamps[1:])
    )
    if tolerance >= minimum_interval / 2:
        raise ValueError("Timestamp shift limit must be less than half the minimum scan interval")
    if max(shifts) > tolerance:
        raise ValueError(
            f"Scan timestamps differ by up to {max(shifts):.9g} seconds; "
            f"allowed shift is {tolerance:.9g} seconds"
        )
    baseline = evaluate(args.baseline, args.reference)
    candidate = evaluate(args.candidate, args.reference)
    changes = {
        key: {
            "baseline": baseline[key],
            "candidate": candidate[key],
            "change": candidate[key] - baseline[key],
            "non_increasing": candidate[key] <= baseline[key],
        }
        for key in METRICS
    }
    passed = all(change["non_increasing"] for change in changes.values())
    report = {
        "baseline_file": str(args.baseline),
        "candidate_file": str(args.candidate),
        "reference_file": str(args.reference),
        "poses": len(baseline_stamps),
        "first_timestamp": baseline_stamps[0],
        "last_timestamp": baseline_stamps[-1],
        "timestamp_comparison": {
            "allowed_shift_seconds": tolerance,
            "maximum_shift_seconds": max(shifts),
            "shifted_scans": sum(shift != 0 for shift in shifts),
            "candidate_first_timestamp": candidate_stamps[0],
            "candidate_last_timestamp": candidate_stamps[-1],
        },
        "sounding_options": {"delta_seconds": 1, "alignment": "Sounding default"},
        "baseline": baseline,
        "candidate": candidate,
        "changes": changes,
        "no_regression": passed,
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(2)
