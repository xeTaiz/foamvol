#!/usr/bin/env bash
# Cooperatively claim and run the 270-arm Stage-3 75-view queue.
set -u

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/code/lc64-radfoam}"
PY="${PY:-python3}"
CFG_DIR_REL="experiments/multiscene_feature_followup_v1/stage3_75_configs"
QUEUE_REL="experiments/multiscene_feature_followup_v1/stage3_75_queue.txt"
SHARED_RUNNER="$HERE/../sweep_revalidate_v1/run_sweep.sh"
ROOT="$REPO/output/multiscene_feature_followup_v1/stage3_75"
CLAIMS="$ROOT/.claims"

cd "$REPO" || exit 1
mkdir -p "$CLAIMS"
export CUDA_VISIBLE_DEVICES="$GPU"

while true; do
    ARM=""
    UNCLAIMED=0
    while IFS= read -r CANDIDATE; do
        [ -n "$CANDIDATE" ] || continue
        if [ -d "$CLAIMS/$CANDIDATE" ]; then
            continue
        fi
        UNCLAIMED=1
        CFG="$CFG_DIR_REL/$CANDIDATE.yaml"
        REF="$($PY -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get("ref_volume_path", ""))' "$CFG")"
        if [ -n "$REF" ] && [ ! -f "$REF" ]; then
            continue
        fi
        if mkdir "$CLAIMS/$CANDIDATE" 2>/dev/null; then
            ARM="$CANDIDATE"
            break
        fi
    done < "$QUEUE_REL"

    if [ -z "$ARM" ]; then
        if [ "$UNCLAIMED" -eq 0 ]; then
            break
        fi
        sleep 20
        continue
    fi

    CFG="$CFG_DIR_REL/$ARM.yaml"
    EXPERIMENT_NAME="$($PY -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment_name"])' "$CFG")"
    RUN="output/$EXPERIMENT_NAME"
    STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    CFG_DIR="$CFG_DIR_REL" REPO="$REPO" PY="$PY" "$SHARED_RUNNER" "$GPU" "$ARM"
    TAG="${ARM#*__}"
    if [ -f "$RUN/DONE" ]; then
        if ! "$PY" "$HERE/assert_stage3_75_active.py" --run "$RUN" --config "$CFG" --tag "$TAG" >>"$RUN/run.log" 2>&1; then
            rm -f "$RUN/DONE"
            touch "$RUN/FAILED"
        fi
    fi
    "$PY" "$HERE/record_provenance.py" --arm "$ARM" --config "$CFG" --run "$RUN" --gpu-index "$GPU" --started-at "$STARTED_AT"
done

printf 'Stage-3 75-view queue drained on %s gpu=%s\n' "$(hostname)" "$GPU"
