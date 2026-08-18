#!/usr/bin/env bash
# Run a queue of sweep_splitcell_v1 arms sequentially on one GPU.
#
#   ./run_sweep.sh <gpu_index> <arm> [arm ...]
#
# Each arm: train -> hard split voxelization (256^3, 4x supersampled) ->
# surface metrics -> continuity metrics (split arms only). Writes DONE or
# FAILED into the run directory so progress is pollable without parsing logs.
#
# Safe to run one instance per GPU concurrently; each arm owns its own output
# directory. Re-running an arm wipes and redoes it.
set -u

GPU="$1"; shift
ARMS=("$@")

REPO=/code/lc64-radfoam
CFG_DIR=experiments/sweep_splitcell_v1/configs
DATA=r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone
GT="$DATA/vol_gt.npy"
PY=/code/lc64-venv/bin/python

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64
export PYTHONPATH="$PWD/src:$PWD"
export MPLBACKEND=Agg

for ARM in "${ARMS[@]}"; do
    CFG="$CFG_DIR/$ARM.yaml"
    RUN="output/sweep_splitcell/$ARM"

    if [ ! -f "$CFG" ]; then
        echo "[$ARM] MISSING CONFIG $CFG" >&2
        continue
    fi

    echo "=== [$ARM] gpu=$GPU start $(date -Is) ==="
    rm -rf "$RUN"
    mkdir -p "$RUN"

    if ! $PY train.py --config "$CFG" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] TRAIN FAILED (see $RUN/run.log)" >&2
        touch "$RUN/FAILED"
        continue
    fi

    # Hard split-aware voxelization: the artifact all headline volume/surface
    # numbers are computed from. Falls back to scalar density automatically
    # for the no-split control, so every arm is scored the same way.
    if ! $PY split_voxelize.py \
            --model "$RUN/model.pt" \
            --resolution 256 --supersample 4 \
            --output "$RUN/volume_hard_ss4.npy" \
            --gt "$GT" \
            --side_map "$RUN/side_hard_ss4.npy" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] VOXELIZE FAILED" >&2
        touch "$RUN/FAILED"
        continue
    fi

    if ! $PY experiments/face_continuity_v1/eval_hard_surface.py \
            --volume "$RUN/volume_hard_ss4.npy" \
            --data_path "$DATA" \
            --output "$RUN/surface_hard_ss4_metrics.json" >>"$RUN/run.log" 2>&1; then
        echo "[$ARM] SURFACE EVAL FAILED" >&2
        touch "$RUN/FAILED"
        continue
    fi

    # Continuity diagnostics need thin-surface state; the scalar control has
    # none, so skip it there rather than logging a spurious failure.
    if [ "$ARM" != "SC256_scalar" ]; then
        if ! $PY experiments/face_continuity_v1/evaluate_continuity.py \
                --model "$RUN/model.pt" \
                --gt "$GT" \
                --output "$RUN/face_continuity_eval.json" >>"$RUN/run.log" 2>&1; then
            echo "[$ARM] CONTINUITY EVAL FAILED" >&2
            touch "$RUN/FAILED"
            continue
        fi
    fi

    touch "$RUN/DONE"
    echo "=== [$ARM] gpu=$GPU done $(date -Is) ==="
done

echo "queue finished on gpu=$GPU"
