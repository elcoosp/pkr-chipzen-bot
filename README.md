# pkr-chipzen-bot

A standalone Rust binary crate that plugs your **pkr-sota** CFR engine into
**chipzen.ai** as an **External-API bot** — the same path as the Python
reference client in `chipzen-sdk/examples/external-api-bot`, but speaking the
protocol natively and deciding with your trained blueprint.

```text
lobby WS ──► matched ──► match gateway WS ──► turn_request ──► pkr-sota
             (resume)       (Sec-WebSocket-         │            blueprint
                            Protocol token)         ▼            lookup
                                              turn_action ◄── CDF sample +
                                                              ladder sizing
```

---

## 1. Layout & wiring to your local pkr-sota

This crate is **standalone** and depends on your local pkr-sota checkout via
path dependencies. Place it as a **sibling** of the engine repo:

```text
your-code/
├── pkr-sota/          ← your engine workspace (with your trained artifacts)
└── pkr-chipzen-bot/   ← this crate
```

`Cargo.toml` points at `../pkr-sota/crates/pkr-*`. If your layout differs,
edit the five `path = …` entries — nothing else changes.

The bot reuses the engine's *runtime* crates only (no training code runs):

| engine crate  | used for                                                |
|---------------|---------------------------------------------------------|
| `pkr-runtime` | `MmapReader` + `SolverHandle::get_advice_fast` (mmap'd `blueprint.bin`, binary-search lookup) |
| `pkr-abstraction` | `KMeansAbstraction::get_infoset_hash` (EHS² cluster + flop bucket + FNV-1a infoset hash), `load_centroids`, `calculate_ehs` (fallback policy) |
| `pkr-eval`    | `TableEvaluator` over `hand_ranks.bin`                  |
| `pkr-contracts` | `SotaAdvice`, `Evaluator`, `AbstractionBuilder` trait  |

---

## 2. Build

```bash
cargo build --release          # binary: target/release/pkr-chipzen-bot
cargo test                     # unit + golden + mock-server integration tests
```

> The engine's `pkr-cfr` (pulled in via `pkr-export`) builds its GPU feature
> by default; it compiles fine on machines without a GPU — it is only
> *used* by the trainer, never by this bot.

---

## 3. Train artifacts (on your Mac), then ship them next to the bot

On the Mac Mini, follow your usual pipeline (`run.sh`): train with
`pkr-trainer` and export `blueprint.bin`. Then copy the artifacts you want to
play with to the machine running the bot:

```text
artifacts/
├── blueprint.bin        # required — mmap'd PKRSOTA1-v2 file
├── hand_ranks.bin       # required — 5-card rank lookup table
├── centroids.bin        # required — EHS² centroid store (bincode)
├── flop_buckets.bin     # optional — flop bucket table
├── flop_centroids.bin   # optional — per-street centroid overrides
├── turn_centroids.bin   # optional
└── river_centroids.bin  # optional
```

**Important — `--ehs-samples` must match training.** The runtime computes
hand clusters by Monte-Carlo EHS² (`pkr-abstraction/src/ehs.rs`), and the
sample count is cached at first use. Your `run.sh` exported
`EHS_SAMPLES=100` for training; the engine default is `1000`. Pass the same
value you trained with:

```bash
pkr-chipzen-bot ... --ehs-samples 100      # if you trained via run.sh
# or: export EHS_SAMPLES=100
```

(Note: your trainer currently passes the full fixed-size board/history arrays
into the hash — see §7 — which makes the *preflop* cluster constant; postflop
clusters depend on EHS sampling, hence the flag.)

---

## 4. Run

```bash
pkr-chipzen-bot \
    --base-url wss://staging.chipzen.ai \
    --bot-id <your-bot-uuid> \
    --token cz_extbot_... \
    --blueprint artifacts/blueprint.bin \
    --rank-table artifacts/hand_ranks.bin \
    --centroids artifacts/centroids.bin \
    --flop-buckets artifacts/flop_buckets.bin \
    [--loop] [-v|-vv|-vvv] [--deterministic] [--ehs-samples 100]
```

Environment fallbacks (CLI flags win, same as the Python client):

| env var                | flag        |
|------------------------|-------------|
| `CHIPZEN_BASE_URL`     | `--base-url` |
| `CHIPZEN_BOT_ID`       | `--bot-id`   |
| `CHIPZEN_EXTBOT_TOKEN` | `--token`    |

Artifact flags (defaults mirror `pkr-trainer`): `--blueprint blueprint.bin
--rank-table hand_ranks.bin --centroids centroids.bin` plus optional
`--flop-buckets`, `--flop-centroids`, `--turn-centroids`,
`--river-centroids`, `--preflop-table`, `--flop-table`, `--turn-table`,
`--river-table`.

Behaviour:

* **One lobby connection per bot** (platform invariant — a second one evicts
  the first with close code 4000, which this bot treats as fatal). The lobby
  socket stays open across matches, unlike the Python reference which
  reconnects per match. A lobby drop triggers exponential-backoff reconnect.
* **Sequential matches**: every `matched` notification (fresh or `resume`
  re-attach) is queued and played to completion, one at a time.
* **`--loop`** keeps playing forever; without it the bot exits after the
  first completed match (like `run.py`).
* **Mid-match reconnect** (§8.6): if the match socket drops, the bot re-dials
  the same gateway URL (budget: 3) and re-runs the handshake; the executor
  re-emits the in-flight `turn_request`.
* **`action_rejected`** is answered with the same `request_id` and a
  guaranteed-legal `check`/`fold` fallback.
* **Never illegal**: every decision is validated against the frame's
  `valid_actions` and the `[min_raise, max_raise]` window before sending;
  `raise` amounts are total bet sizes; there is no `all_in` string (shoving =
  `raise` to `max_raise`).

---

## 5. Decision pipeline

For every `turn_request` (stateless — everything is derived from the frame):

1. **Replay** `state.action_history` into the trainer's 6 abstract buckets
   (fold / check-call / small / medium / large bet / all-in), tracking pot +
   per-seat money exactly like `pkr-core::GameState` so each raise is
   bucketed with the engine's *post-action pot* fraction semantics. Blinds
   and antes update money but are not bucketed (trainer behaviour).
2. **Hash** the infoset with `KMeansAbstraction::get_infoset_hash` using the
   trainer's exact inputs: your hole cards, the zero-padded 5-byte board, the
   zero-padded 32-byte bucket history, street code.
3. **Look up** the hash in the blueprint (`get_advice_fast`).
4. **Sample** the CDF (`--deterministic` = argmax instead).
5. **Map** the abstract slot to a wire action:
   * slot 0 → `fold` (degrade to check/call when folding isn't offered)
   * slot 1 → `check`/`call`
   * slots 2–4 → `raise` sized by **inverting the trainer's own ladder**
     (`0.5/0.75/1.0/1.5/2.0 × pot`, the sizes `legal_actions()` generated in
     training), clamped to `[min_raise, max_raise]`, choosing the candidate
     whose recomputed bucket equals the sampled slot; never escalating to a
     shove unless the blueprint said so
   * slot 5 → `raise` to `max_raise` (shove), or call when raising isn't
     available
6. **Fallback on blueprint miss**: an EHS heuristic — fold when equity is far
   below pot odds (EHS < max(0.30, required − 0.03)), call when close, raise
   ~1× pot when EHS ≥ 0.70, open-bet 0.5–1× pot when EHS ≥ 0.62/0.85.

---

## 6. Tests (TDD)

```bash
cargo test
```

* **unit tests** — card encoding, frame serde (documented examples from the
  protocol doc), bucket replay (golden hand from the poker-protocol §4
  example), CDF sampling/argmax, ladder sizing, legality validation, config
  checks, backoff/masking.
* **`tests/golden.rs`** — pins the trainer-exact hashing behaviour,
  *including the documented quirks* (padded 32-byte history, padded 5-byte
  board, constant preflop cluster). If you retrain with a fixed trainer,
  update these alongside.
* **`tests/integration.rs`** — a mock chipzen platform (tokio-tungstenite)
  runs a full match against the real bot: lobby auth + heartbeat + matched,
  match handshake with token-in-subprotocol assertion, four turn_requests
  (check / call / shove / miss→fallback), an `action_rejected` retry with the
  same `request_id`, and a clean `match_end`. Deterministic synthetic
  artifacts (all-zero rank table ⇒ EHS ≡ 0.5) make every expected action
  exact.

---

## 7. Blueprint-compatibility contract (do not "fix" casually)

`src/adapter.rs` replicates the *trainer's* hash inputs byte-for-byte as
found in the pkr-sota repo dump:

1. `traversal.rs` builds `history_bytes` from the fixed `[u8; 32]`
   `abstract_history` array (using its `.len()`, i.e. always 32) and passes
   all 32 bytes — buckets + zero padding.
2. It passes `&current.board` — the fixed `[u8; 5]` array — so the board is
   always 5 bytes with zero padding (preflop = five zero bytes).
3. Consequence: preflop EHS evaluations degenerate to a tie
   (3 unique cards < 5), so the preflop cluster is constant and preflop
   infosets are history-only. `tests/golden.rs` pins this.

Because the bot path-depends on `pkr-abstraction` and only feeds it the same
inputs, any future engine change that fixes these quirks automatically
requires retraining — and then only the golden tests (and a fresh blueprint)
change, not the bot.

---

## 8. Operational notes

* Decision latency is dominated by postflop EHS sampling (~ms with the table
  evaluator); far under the 30 s agent clock.
* Bot→server frames are tiny; the 4096-byte cap is never approached.
* Logs go to stderr via `tracing` (`-v` debug, `-vv` trace, `RUST_LOG`
  overrides). The token is always masked.
* Exit codes: `0` after a completed match (or clean loop shutdown), `1` on
  fatal errors (bad config, evict, auth failure).
* Keep exactly one instance per bot; scale by creating more bots/tokens.

## 9. Known limitations

* Heads-up only (the platform's remote path is HU-only today, §6.2 of the
  protocol doc); the state parser rejects `num_players != 2` inputs
  implicitly via the two-seat replay.
* Antes are folded into the pot as dead money (the engine has no ante
  concept); chipzen defaults `ante = 0`.
* The engine's runtime crates are reused as-is; blueprint quirks (§7) are
  replicated, not fixed.
