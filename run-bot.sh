#!/usr/bin/env bash
# pkr-chipzen-bot — best-config production launcher (k7-baseline).
#
# Encodes the shipped pkr-sota runtime configuration as measured in the
# engine's own experiment docs (see README.md "Best configuration" for the
# evidence and the full rationale).
set -euo pipefail
cd "$(dirname "$0")"

: "${CHIPZEN_BASE_URL:=wss://staging.chipzen.ai}"
: "${CHIPZEN_BOT_ID:?set CHIPZEN_BOT_ID (the external-API bot UUID)}"
: "${CHIPZEN_EXTBOT_TOKEN:?set CHIPZEN_EXTBOT_TOKEN (cz_extbot_...)}"

BIN=target/release/pkr-chipzen-bot
if [ ! -x "$BIN" ]; then
  echo "binary missing — run ./scripts/build.sh first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# pkr-sota runtime configuration — the shipped, measured best
# ---------------------------------------------------------------------------
# The k7-baseline checkpoint (K=7 action buckets, signature v5, 100 bb,
# eps=0.01, loose min-raise) is the engine's shipped production checkpoint
# (docs/handoffs/HANDOFF_2026-10-12_session9.md; run1-combined was graded
# DO NOT SHIP).
export PKR_STACK_BB=100        # checkpoint is 100 bb; the fingerprint pins it

# Signature generation: the checkpoint was trained with PKR_SIG_V5=1.
# The binary defaults to --sig 5; passed explicitly below for auditability.

# ---------------------------------------------------------------------------
# Session mode (optional): river-only bounded subgame search
# ---------------------------------------------------------------------------
# Pass PKR_CKPT=artifacts/train.ckpt to enable RuntimeSession: opponent
# range tracking + river-only vsearch. PKR_VSEARCH=1 selects the PROVEN
# heuristic bound (the shipped safe search config; H2H vs bare blueprint
# +0.958 t=5.28 / +1.279 t=7.02 on k7). River decision ≈ 0.7 s in the
# 2000 ms budget; the 30 s agent clock is comfortable.
if [ -n "${PKR_CKPT:-}" ]; then
  export PKR_VSEARCH=1
  export PKR_VSEARCH_BUDGET_MS="${PKR_VSEARCH_BUDGET_MS:-2000}"
fi

# Deliberately NOT set — every one of these is a measured regression or an
# offline-only feature (see docs/experiments in pkr-sota):
#   PKR_VSEARCH_GADGET / PKR_VSEARCH_RAW   opt-outs from the proven bound
#   PKR_VSEARCH_STREETS                    turn search ≈45–52 s/decision — OFFLINE ONLY
#   PKR_STRICT_MIN_RAISE                   k7 trained with loose rules; ON changes legal_actions
#   PKR_MOMENTUM / PKR_TRANSLATE / PKR_SHADOW / PKR_BUCKET_U16
#   PKR_ALLOW_EHS_FALLBACK                 trainer-only guard

# ---------------------------------------------------------------------------
# Artifacts (produced by the engine pipeline, see artifacts/README.md)
# ---------------------------------------------------------------------------
A=artifacts
ARGS=(
  --base-url "$CHIPZEN_BASE_URL"
  --bot-id "$CHIPZEN_BOT_ID"
  --token "$CHIPZEN_EXTBOT_TOKEN"
  --blueprint "$A/blueprint.bin"
  --centroids "$A/centroids.bin"
  --flop-table "$A/abstraction.bin"
  --turn-table "$A/turn_abstraction.bin"
  --river-table "$A/river_buckets.bin"
  --preflop-k 200
  --sig 5
)
[ -n "${PKR_CKPT:-}" ] && ARGS+=(--ckpt "$PKR_CKPT")
# Optional tables — loaded for exact artifact parity with the trainer when
# present (the engine warns if its hash path ever needs a missing one).
[ -f "$A/preflop_abstraction.bin" ] && ARGS+=(--preflop-table "$A/preflop_abstraction.bin")
[ -f "$A/flop_buckets.bin" ] && ARGS+=(--flop-buckets "$A/flop_buckets.bin")
[ -f "$A/hand_ranks.bin" ] && ARGS+=(--rank-table "$A/hand_ranks.bin")
[ -f "$A/flop_centroids.bin" ] && ARGS+=(--flop-centroids "$A/flop_centroids.bin")
[ -f "$A/turn_centroids.bin" ] && ARGS+=(--turn-centroids "$A/turn_centroids.bin")
[ -f "$A/river_centroids.bin" ] && ARGS+=(--river-centroids "$A/river_centroids.bin")

#Behaviour toggles.
[ "${LOOP:-0}" = "1" ] && ARGS+=(--loop)
[ "${DETERMINISTIC:-0}" = "1" ] && ARGS+=(--deterministic)
[ -n "${EHS_SAMPLES:-}" ] && ARGS+=(--ehs-samples "$EHS_SAMPLES")
[ "${FALLBACK_EHS:-0}" = "1" ] && ARGS+=(--fallback-ehs)

exec "$BIN" "${ARGS[@]}" "$@"
