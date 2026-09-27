#!/usr/bin/env sh
# Usage: ./run.sh demo | tests | experiments [--quick]
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
case "$1" in
  demo) solara run app.py ;;
  tests) shift; python -m pytest "$@" ;;
  experiments)
    if [ "$2" = "--quick" ]; then python run_experiments.py --quick
    else python run_experiments.py --scenarios all --runs 30 && python run_studies.py; fi ;;
  *) echo "usage: ./run.sh demo | tests | experiments [--quick]"; exit 1 ;;
esac
