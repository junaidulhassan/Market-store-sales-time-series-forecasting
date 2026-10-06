#!/usr/bin/env bash
# Start the React (Vite) dashboard.
set -e
cd "$(dirname "$0")/frontend"
npm install
npm run dev
