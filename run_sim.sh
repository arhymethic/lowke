#!/usr/bin/env bash
# Quick root launcher for Highway CCTV Simulator
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
exec "$DIR/sim/run.sh" "$@"
