#!/usr/bin/env bash
# Run one randomized Stage-2A scene/seed block sequentially on one GPU.
set -eu

GPU="$1"
shift
ARMS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/code/lc64-radfoam}"
PY="${PY:-python3}"
CFG_DIR_REL="experiments/multiscene_feature_followup_v1/stage2a_configs"
SHARED_RUNNER="$HERE/../sweep_revalidate_v1/run_sweep.sh"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"

for ARM in "${ARMS[@]}"; do
    CFG="$CFG_DIR_REL/$ARM.yaml"
    if [ ! -f "$CFG" ]; then
        echo "[$ARM] MISSING CONFIG $CFG" >&2
        exit 1
    fi
    EXPERIMENT_NAME="$($PY -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment_name"])' "$CFG")"
    RUN="output/$EXPERIMENT_NAME"
    STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    CFG_DIR="$CFG_DIR_REL" REPO="$REPO" PY="$PY" \
        "$SHARED_RUNNER" "$GPU" "$ARM"

    TAG="${ARM#*__}"
    TAG="${TAG%_s[0-9]*}"
    if [ -f "$RUN/DONE" ]; then
        if ! "$PY" "$HERE/assert_stage2a_active.py" \
                --run "$RUN" --config "$CFG" --tag "$TAG" >>"$RUN/run.log" 2>&1; then
            rm -f "$RUN/DONE"
            touch "$RUN/FAILED"
        fi
    fi

    "$PY" "$HERE/record_provenance.py" \
        --arm "$ARM" \
        --config "$CFG" \
        --run "$RUN" \
        --gpu-index "$GPU" \
        --started-at "$STARTED_AT"
done

printf 'Stage-2A block finished on %s gpu=%s arms=%s\n' "$(hostname)" "$GPU" "${#ARMS[@]}"
