#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || { echo 'Build on a Linux ARM64 DGX Spark.' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
exec docker build --platform linux/arm64 -f "$here/Dockerfile" -t "${INKLING_IMAGE:-tensorfold-inkling2:spark}" "$root"
