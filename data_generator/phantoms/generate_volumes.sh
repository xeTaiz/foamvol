#!/usr/bin/env bash
# Generate all three debug phantom volumes.
# Run from the radfoam repo root:
#   micromamba run -n radfoam bash data_generator/phantoms/generate_volumes.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Shepp-Logan ==="
python "$SCRIPT_DIR/shepp_logan_3d.py"

echo "=== NEMA IEC ==="
python "$SCRIPT_DIR/nema_iec.py"

echo "=== Marschner-Lobb ==="
python "$SCRIPT_DIR/marschner_lobb.py"

echo "All volumes written to /mnt/hdd/r2_data/debug_phantoms/volumes/"
