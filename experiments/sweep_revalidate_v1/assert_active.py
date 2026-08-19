#!/usr/bin/env python3
"""Activation guard for sweep_revalidate_v1 arms.

The 13-arm split-cell sweep burned every regularized arm because
face_continuity/candidate_faces was 0.0 for the entire run and nobody
checked until afterwards. This script closes that gap: for every arm whose
manifest entry names a TensorBoard `assert_scalar` tag, it requires that tag
to exist in the run's event file and to have taken at least one nonzero
value, then writes an ACTIVE or INACTIVE marker into the run directory.

Exit code is nonzero exactly when the arm is INACTIVE (or the check itself
could not run), so run_sweep.sh can log the failure -- but run_sweep.sh
deliberately does NOT `continue`/discard the run on that exit: an inactive
arm is a finding, not a crash.

Usage:
    python experiments/sweep_revalidate_v1/assert_active.py --run RUN_DIR --config CFG
"""

import argparse
import glob
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.yaml")


def load_manifest():
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def find_arm(manifest, tag):
    for arm in manifest["arms"]:
        if arm["tag"] == tag:
            return arm
    return None


def _write_marker(run_dir, state):
    # Clear any marker from a previous attempt so stale state can't linger.
    for f in ("ACTIVE", "INACTIVE", "UNKNOWN"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    open(os.path.join(run_dir, state), "w").close()


def _grep_loss_fallback(run_dir, tag):
    """Fall back to run.log if no TensorBoard event file exists at all."""
    log_path = os.path.join(run_dir, "run.log")
    if not os.path.exists(log_path):
        return None
    loss_name = tag.rsplit("/", 1)[-1]
    pattern = re.compile(re.escape(loss_name) + r"[^0-9eE.+-]*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    values = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    values.append(float(m.group(1)))
                except ValueError:
                    pass
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run output directory")
    parser.add_argument("--config", required=True, help="Path to the arm's config YAML")
    args = parser.parse_args()

    run_dir = args.run
    tag = os.path.splitext(os.path.basename(args.config))[0]

    manifest = load_manifest()
    arm = find_arm(manifest, tag)
    if arm is None:
        print(f"[assert_active] WARNING: arm '{tag}' not found in manifest.yaml; "
              f"treating as no assert_scalar (ACTIVE).")
        _write_marker(run_dir, "ACTIVE")
        return 0

    scalar = arm.get("assert_scalar")
    if not scalar:
        _write_marker(run_dir, "ACTIVE")
        print(f"[assert_active] {tag}: assert_scalar is null -> ACTIVE")
        return 0

    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        values = _grep_loss_fallback(run_dir, scalar)
        if values is None:
            print(f"[assert_active] {tag}: no event file and no run.log; UNKNOWN")
            _write_marker(run_dir, "UNKNOWN")
            return 1
        if not values or not any(abs(v) > 0 for v in values):
            print(f"[assert_active] {tag}: no event file; run.log fallback found "
                  f"{len(values)} values, none nonzero -> INACTIVE")
            _write_marker(run_dir, "INACTIVE")
            return 1
        values_sorted = sorted(values)
        mid = values_sorted[len(values_sorted) // 2]
        print(f"[assert_active] {tag}: no event file; run.log fallback for '{scalar}': "
              f"min={min(values):.6g} median={mid:.6g} max={max(values):.6g} -> ACTIVE")
        _write_marker(run_dir, "ACTIVE")
        return 0

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()

    if scalar not in ea.Tags().get("scalars", []):
        print(f"[assert_active] {tag}: tag '{scalar}' not found in event file -> INACTIVE")
        _write_marker(run_dir, "INACTIVE")
        return 1

    events = ea.Scalars(scalar)
    values = [e.value for e in events]
    if not values or not any(abs(v) > 0 for v in values):
        print(f"[assert_active] {tag}: tag '{scalar}' present but all-zero "
              f"({len(values)} steps) -> INACTIVE")
        _write_marker(run_dir, "INACTIVE")
        return 1

    values_sorted = sorted(values)
    mid = values_sorted[len(values_sorted) // 2]
    print(f"[assert_active] {tag}: tag '{scalar}' min={min(values):.6g} "
          f"median={mid:.6g} max={max(values):.6g} ({len(values)} steps) -> ACTIVE")
    _write_marker(run_dir, "ACTIVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
