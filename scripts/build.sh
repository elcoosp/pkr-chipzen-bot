#!/usr/bin/env bash
# Build the bot (release). Requires the pkr-sota checkout as a sibling:
#   your-code/pkr-sota/  (engine workspace)
#   your-code/pkr-chipzen-bot/  (this crate)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f ../pkr-sota/Cargo.toml ]; then
  echo "error: ../pkr-sota not found — the bot path-depends on the engine checkout" >&2
  echo "       clone it: git clone https://github.com/elcoosp/pkr-sota ../pkr-sota" >&2
  exit 1
fi

cargo build --release -p pkr-chipzen-bot
echo
echo "binary: $(pwd)/target/release/pkr-chipzen-bot"
echo "tests:  cargo test -p pkr-chipzen-bot"
