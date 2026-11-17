#!/usr/bin/env bash
# Launch go2_ctrl with onnxruntime on LD_LIBRARY_PATH (doggg layout).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib:${LD_LIBRARY_PATH:-}"
exec "${ROOT}/deploy/robots/go2/build/go2_ctrl" "$@"
