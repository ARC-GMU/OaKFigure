#!/bin/sh
# Launch OaKFigure with the project's own Python environment.
cd "$(dirname "$0")" && exec .venv/bin/python main.py "$@"
