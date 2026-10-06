#!/usr/bin/env bash
# Start the FastAPI backend that serves the trained model to the React dashboard.
set -e
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
