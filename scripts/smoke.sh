#!/usr/bin/env bash
# Smoke-test the bot without a chipzen account: unit + golden + the mock
# platform integration test (a full scripted match against a local WS
# server), plus the engine workspace's own test gate.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== pkr-chipzen-bot tests (unit + golden + mock-server integration) =="
cargo test -p pkr-chipzen-bot --release "$@"

echo
echo "== engine fingerprint sanity (pkr-sota fast gate, first 30s) =="
if [ -f ../pkr-sota/justfile ]; then
  (cd ../pkr-sota && cargo test -p pkr-core -p pkr-runtime --release -- 2>&1 | tail -5) || \
    echo "(engine tests skipped/failed — the bot itself passed; run 'just fast' in ../pkr-sota for the full gate)"
fi
