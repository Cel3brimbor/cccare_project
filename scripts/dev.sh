#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
if [[ ! -d frontend/node_modules ]]; then
  (cd frontend && npm install)
fi
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
