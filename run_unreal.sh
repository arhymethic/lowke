#!/usr/bin/env bash
# Quick root launcher for Unreal Multi-Camera Re-ID Pipeline
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
exec "$DIR/Unreal/run.sh" "$@"
