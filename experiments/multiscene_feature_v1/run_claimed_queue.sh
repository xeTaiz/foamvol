#!/usr/bin/env bash
# Concurrent multi-scene queue: every worker claims one arm atomically on the
# shared /code filesystem, then delegates its complete train/evaluate path to
# run_sweep.sh. Reference-guided arms are withheld until that scene's BASE_s42
# hard volume is available.
set -u

GPU="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/code/lc64-radfoam}"
PY="${PY:-python3}"
MANIFEST="$REPO/experiments/multiscene_feature_v1/arms.json"
ROOT="$REPO/output/multiscene_feature_v1"
CLAIMS="$ROOT/.claims"

cd "$REPO" || exit 1
mkdir -p "$CLAIMS"

while ARM="$(
    "$PY" - "$MANIFEST" "$ROOT" "$CLAIMS" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest, root, claims = map(Path, sys.argv[1:])
arms = json.loads(manifest.read_text())
extra_families = {"PRUNE_GRID", "DENS_GRID", "REFG_GRID"}


def priority(arm: dict[str, object]) -> tuple[int, str, int, str]:
    tag = str(arm["tag"])
    family = str(arm["family"])
    if family == "BASE":
        return (0, str(arm["scene"]), int(arm["seed"]), tag)
    if family in extra_families and not bool(arm["needs_reference"]):
        return (1, str(arm["scene"]), 0, tag)
    if not bool(arm["needs_reference"]):
        return (2, str(arm["scene"]), 0, tag)
    if family in extra_families:
        return (3, str(arm["scene"]), 0, tag)
    return (4, str(arm["scene"]), 0, tag)

for arm in sorted(arms, key=priority):
    if arm["family"] == "SMOKE":
        continue
    scene, tag = str(arm["scene"]), str(arm["tag"])
    run = root / scene / tag
    if (run / "DONE").is_file() or (run / "FAILED").is_file():
        continue
    if bool(arm["needs_reference"]) and not (root / scene / "BASE_s42" / "volume_hard_ss4.npy").is_file():
        continue
    claim = claims / f"{scene}__{tag}"
    try:
        claim.mkdir()
    except FileExistsError:
        continue
    (claim / "owner").write_text(f"{os.uname().nodename} {os.getpid()}\n")
    print(f"{scene}__{tag}")
    break
PY
)"; do
    [ -n "$ARM" ] || break
    echo "=== [$ARM] claimed by $(hostname) gpu=$GPU $(date -Is) ==="
    "$HERE/run_sweep.sh" "$GPU" "$ARM"
done

echo "claimed queue finished on $(hostname) gpu=$GPU"
