#!/usr/bin/env bash
# Run a queue of sweep_revalidate_v1 arms sequentially on one GPU.
#
#   ./run_sweep.sh <gpu_index> <arm> [arm ...]
#
# Each arm: train -> hard SS4 voxelization (256^3, centre-registered) ->
# eval_vol.py (JSON) -> air_metrics.py -> assert_active.py. Writes DONE or
# FAILED into the run directory so progress is pollable without parsing
# logs. A non-zero exit from assert_active.py does NOT count as a failure:
# an inactive arm is a finding, not a crash, so the run is still marked
# DONE with an INACTIVE marker alongside it.
#
# Safe to run one instance per GPU concurrently; each arm owns its own
# output directory. Re-running an arm wipes and redoes it.
set -u

GPU="$1"; shift
ARMS=("$@")

REPO="${REPO:-/code/lc64-radfoam}"
CFG_DIR="${CFG_DIR:-experiments/sweep_revalidate_v1/configs}"
PY="${PY:-/code/lc64-venv/bin/python}"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64
export PYTHONPATH="$PWD/src:$PWD"
export MPLBACKEND=Agg

for ARM in "${ARMS[@]}"; do
    CFG="$CFG_DIR/$ARM.yaml"

    if [ ! -f "$CFG" ]; then
        echo "[$ARM] MISSING CONFIG $CFG" >&2
        continue
    fi
    EXPERIMENT_NAME="$(
        $PY -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment_name"])' "$CFG"
    )"
    case "$EXPERIMENT_NAME" in
        ""|/*|*..*)
            echo "[$ARM] INVALID experiment_name $EXPERIMENT_NAME" >&2
            continue
            ;;
    esac
    RUN="output/$EXPERIMENT_NAME"

    DATA="$(
        $PY -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["data_path"])' "$CFG"
    )"
    GT="$DATA/vol_gt.npy"
    if [ ! -f "$GT" ]; then
        echo "[$ARM] MISSING GT $GT" >&2
        mkdir -p "$RUN"
        touch "$RUN/FAILED"
        continue
    fi
    if ! RESOLUTION="$(
        $PY -c 'import numpy as np, sys; shape = np.load(sys.argv[1], mmap_mode="r").shape; assert len(shape) == 3 and len(set(shape)) == 1, f"expected cubic GT, got {shape}"; print(shape[0])' "$GT"
    )"; then
        echo "[$ARM] INVALID GT GRID $GT" >&2
        mkdir -p "$RUN"
        touch "$RUN/FAILED"
        continue
    fi

    echo "=== [$ARM] gpu=$GPU start $(date -Is) gt=$GT resolution=$RESOLUTION ==="
    rm -rf "$RUN"
    mkdir -p "$RUN"

    if ! $PY train.py --config "$CFG" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] TRAIN FAILED (see $RUN/run.log)" >&2
        touch "$RUN/FAILED"
        continue
    fi

    # Centre-registered hard SS4 voxelization at the GT's cubic resolution.
    # This makes the evaluator's equal-grid contract explicit for every scene.
    if ! $PY split_voxelize.py \
            --model "$RUN/model.pt" \
            --resolution "$RESOLUTION" --supersample 4 \
            --output "$RUN/volume_hard_ss4.npy" \
            --gt "$GT" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] VOXELIZE FAILED" >&2
        touch "$RUN/FAILED"
        continue
    fi

    if ! $PY eval_vol.py "$RUN/volume_hard_ss4.npy" "$GT" \
            --json "$RUN/eval_vol.json" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] EVAL_VOL FAILED" >&2
        touch "$RUN/FAILED"
        continue
    fi

    if ! $PY air_metrics.py \
            --prediction "$RUN/volume_hard_ss4.npy" \
            --gt "$GT" \
            --output "$RUN/air_metrics.json" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] AIR_METRICS FAILED" >&2
        touch "$RUN/FAILED"
        continue
    fi

    # Multi-scene configs use <scene>__<source-tag> filenames. The source
    # tag retains its activation assertion from the authoritative manifest.
    ASSERT_TAG="${ARM##*__}"
    $PY experiments/sweep_revalidate_v1/assert_active.py \
            --run "$RUN" --config "$CFG" --tag "$ASSERT_TAG" >>"$RUN/run.log" 2>&1

    touch "$RUN/DONE"
    echo "=== [$ARM] gpu=$GPU done $(date -Is) ==="
done

echo "queue finished on gpu=$GPU"
