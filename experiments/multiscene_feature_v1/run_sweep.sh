#!/usr/bin/env bash
# Run one sequential multi-scene queue. Interface matches the shared runner:
# ./run_sweep.sh <gpu_index> <scene__arm> [scene__arm ...]
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="experiments/multiscene_feature_v1/configs" \
    exec "$HERE/../sweep_revalidate_v1/run_sweep.sh" "$@"
