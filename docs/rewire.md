# pkr-chipzen-bot ↔ pkr-sota Re-Wire Plan (agent-executable)

**Goal**: make `pkr-chipzen-bot` correctly use the **current** `pkr-sota` implementation.
The pkr-sota repo did **not** move (same path, same crate layout — the bot's path
dependencies still resolve). What moved is the **implementation**: the infoset-hash
inputs, the action ladder/buckets, the EHS return type, and the blueprint file format.

**Source of truth** (verified by reading source, not docs):
- pkr-sota dump: `upload/dump.txt` (evidence anchors cited as `SOTA L<n>`)
- chipzen dump: `upload/dump-chipzen.txt` (anchors `CZ L<n>`)

**Audience**: an autonomous agent with no context. Follow tasks **in order**. Copy all
code blocks **verbatim**. Do not refactor anything not named in a task.

---

## 0. Non-negotiable rules for the executing agent

1. Execute TASK 1 → TASK 7 in order. Never skip a gate.
2. Only edit files listed under "You may touch". **Never edit anything under
   `../pkr-sota/`** (read-only dependency — you only *build* and *run* it in TASK 1).
3. After each task, run its Gate. If a gate fails: undo your last edit
   (`git checkout -- <file>` or revert manually), re-apply it exactly, and if it still
   fails, STOP and report the exact error output.
4. One logical change per commit. Commit message format: `T<n>: <one-line summary>`.
5. If a code block in this plan does not match the file (e.g. it was already edited),
   STOP and report the mismatch instead of improvising.
6. Final acceptance = TASK 5 + TASK 6 gates green.

---

## 1. What actually moved (ground truth)

Four implementation changes break the bot. Everything else in the pkr-sota API
surface the bot touches is **unchanged** (see Appendix C).

### Change 1 — `calculate_ehs` now returns `(f32, f32)` → bot does not compile
- OLD: returned `(f64, f64)`.
- NEW: `pub fn calculate_ehs(hole: &[u8], board: &[u8], evaluator: &dyn Evaluator) -> (f32, f32)`
  (SOTA `crates/pkr-abstraction/src/ehs.rs`, anchor L1576–1625).
- Bot breakage: `src/strategy.rs` `fallback_decision` does
  `let equity = calculate_ehs(...).0;` then `equity >= 0.62` (f64 literal) →
  type error E0308 with an f32 left side (CZ L2711–2727).

### Change 2 — Infoset-hash history input: 32-byte bucket history → 4-byte signature
- OLD (what the bot reproduces): the abstract-bucket history zero-padded to 32 bytes,
  fed to `get_infoset_hash` (`CZ adapter.rs padded_history()`, L526–532).
- NEW: the trainer hashes a **4-byte little-endian `u32`** produced by
  `GameState::history_signature()`:
  ```text
  bits 0..7   actions_this_street   (fold/check/call/raise count in current street)
  bits 8..15  raises                (Bet actions across the WHOLE hand so far)
  bit  16     last_was_bet          (1 iff the most recent action was a Bet)
  ```
  Evidence: SOTA `crates/pkr-core/src/state.rs::history_signature` (L6198–6216) and
  `crates/pkr-cfr/src/traversal.rs` (L4897–4907):
  ```rust
  let sig = current.history_signature();
  let history_bytes = sig.to_le_bytes();          // 4 bytes
  let infoset_hash = abstraction.get_infoset_hash(hole, board, &history_bytes, street_code);
  ```
- The abstraction crate then hashes `street || history.len() || history || cluster_id ||
  flop_bucket` with FNV-1a (SOTA `crates/pkr-abstraction/src/lib.rs` L1849–1929).
  The bot calls the same function via the `AbstractionBuilder` trait, so it inherits
  the new format **only if it feeds the same 4-byte signature**.
- Bot breakage if unfixed: every hash misses the blueprint → `advice_for` always
  `None` → the bot plays the weak EHS fallback policy 100% of the time.

### Change 3 — Bet ladder and abstract buckets changed
- NEW legal actions per node (`pkr-core/src/state.rs::legal_actions_into`, L6130–6190):
  bet/raise fractions are **`[0.5, 1.0, 2.0]`** + an all-in candidate (OLD ladder was
  `[0.5, 0.75, 1.0, 1.5, 2.0]` + all-in). Plus a new raise cap
  `MAX_RAISES_PER_STREET = 3` (L6139).
- NEW bucket mapping (`pkr-cfr/src/traversal.rs::abstract_action_index`, L5060–5078,
  called with the **PRE-action** state):
  ```rust
  fold → 0;  check/call → 1;
  Bet(amount): if amount >= stack + street_bet            → 5 (all-in)
               else let fraction = amount / max(pot, 1.2);
               if fraction < 1.5 → 2
               else if fraction < 1.2 → 3      // DEAD branch (unreachable), kept verbatim
               else → 4
  ```
  Key differences vs the bot's current math (`CZ adapter.rs::bucket_of_raise`
  L661–676 and `replay_abstract_history` L617–638):
  - fraction uses the **PRE-action pot** with a **1.2 floor** (old: post-action pot,
    1.0 floor);
  - the size threshold is **1.5** (old: 0.5 / 1.0 two-tier);
  - **bucket 3 is unreachable** in the engine (dead branch).
- Bot breakage if unfixed: `ladder_sized_raise`/`map_slot` emit raise sizes that map
  to different buckets than during training → decisions land off-tree.

### Change 4 — Blueprint file format V2 + hash-algo guard
- `MmapReader::new` now **rejects** old blueprints: requires `version >= 2`
  (`FORMAT_VERSION_V2`) and `hash_algo == HASH_ALGO_FNV1A64_INFOSET (= 2)`,
  magic `PKRSOTA1` (SOTA `crates/pkr-runtime/src/mmap.rs` L9145–9169,
  `crates/pkr-export/src/header.rs` L8016).
- Writer emits V2 + FNV + sorted keys + K=6 CDF quantized `(cumulative*255).round()`
  (SOTA `crates/pkr-export/src/writer.rs` L8383–8424).
- Consequence: **any pre-existing `blueprint.bin` must be retrained/re-exported with
  the new trainer** (TASK 1). `pick_slot`'s sampling math (draw `r ∈ [0,255)`) is still
  exact — unchanged, do not touch.

### Unchanged (verified — do NOT touch)
- Crate layout & names: `crates/pkr-{contracts,core,eval,abstraction,runtime,export}`
  (SOTA root `Cargo.toml` L64–77). Bot's path deps in `Cargo.toml` stay as-is.
- `pkr_runtime::{SolverHandle, MmapReader}`; `SolverHandle::new`, `.get_advice_fast`,
  `MmapReader::new`, `.file_header()` fields `infoset_count / max_actions_k / version`
  (SOTA `crates/pkr-runtime/src/{lib,lookup,mmap}.rs`).
- `pkr_contracts::{SotaAdvice { cdf_probabilities: [u8;16], len }, AbstractionBuilder}`
  (SOTA L5634–5650).
- `pkr_abstraction::{load_centroids, KMeansAbstraction::{from_store,
  load_street_centroids, load_flop_buckets, init_table, get_infoset_hash}}` —
  identical signatures (SOTA L1647–1721, L1849–1851).
- `pkr_eval::TableEvaluator::new` + `impl Evaluator for TableEvaluator`
  (SOTA L6575–6652).
- Board input = the trainer's **fixed 5-byte zero-padded array** (traversal passes
  `&current.board`, a `[u8;5]`) → the bot's `padded_board()` stays correct.
- Cluster path: because the board slice is always 5 bytes, the hash always takes the
  `board.len() == 5` river arm: `cluster_id = (hand_rank >> 6) << 8 | river_table[board_bucket]`,
  mixed with a `flop_buckets` byte (SOTA L1898–1928). Therefore, of the optional
  artifacts, **only `--river-table` and `--flop-buckets` influence the hash** — but the
  bot must pass the same set the trainer used (TASK 7).
- Trainer artifact-loading order == bot `Engine::load` order (compare SOTA trainer
  `main.rs` L527–557 with CZ `strategy.rs` L2620–2657). No change needed.
- rand 0.10 / bytemuck 1 / memmap2 0.9 workspace deps — compatible with the bot's
  `Cargo.toml` (SOTA root Cargo.toml L85–109).

---

## 2. Prerequisites (check before TASK 1)

```bash
# From the bot repo root (the folder containing src/ and Cargo.toml):
ls ../pkr-sota/Cargo.toml Cargo.toml   # both must exist; STOP if not
rustc --version && cargo --version
git status --porcelain                  # must be clean; commit or stash first
```

Useful log line for later (already in the bot, CZ L2661–2666): after loading, the bot
logs `blueprint loaded` with `infosets`, `max_actions_k`, `version`. A correct new
blueprint prints `version=2, max_actions_k=6`.

---

## TASK 1 — Rebuild pkr-sota and regenerate artifacts: important  DO NOT DO IT IT'S ALREADY DONE artifacts are ok

**Why first**: old blueprints are hard-rejected by the new loader (Change 4), and the
new hash inputs (TASK 2) only match keys produced by the new trainer.

**You may touch**: nothing in git — this task only builds and runs `../pkr-sota`
and writes artifacts under `../pkr-sota/outputs/`.

Steps (from inside `../pkr-sota`):

```bash
cd ../pkr-sota
cargo build --release -p pkr-trainer -p pkr-abstraction
# Full pipeline (hand_ranks → centroids → buckets → street tables → TRAIN).
# For a first smoke, train short; retrain long later:
ITERATIONS=20000 VERSION=smoke ./run.sh
```

`run.sh` (SOTA L15678–15748) performs exactly 8 steps and writes into
`outputs/<VERSION>/`: `hand_ranks.bin`, `centroids.bin`, `flop_buckets.bin`,
`river_buckets.bin`, `preflop_abstraction.bin`, `abstraction.bin` (flop table),
`turn_abstraction.bin`, `blueprint.bin` (+ `metrics.csv`, `stats.json`, `train.ckpt`).

Gate 1:
```bash
ls -la outputs/smoke/blueprint.bin && head -c 8 outputs/smoke/blueprint.bin   # expect: PKRSOTA1
```
Also record the artifact list — the bot must later be started with the SAME optional
artifact flags the trainer used (see TASK 7 "run command").

If `run.sh` cannot run in your environment, run the 8 `cargo run ... precompute ...`
commands and the final `cargo run --release -p pkr-trainer -- ...` from `run.sh`
verbatim (copy them out of `../pkr-sota/run.sh`).

---

## TASK 2 — `src/adapter.rs`: 32-byte bucket replay → 4-byte history signature

**You may touch**: `src/adapter.rs` only.

### 2.1 `TurnContext` field swap
Replace the field (CZ L463–465):
```rust
    /// Abstract action buckets replayed from `action_history`
    /// (meaningful prefix; zero-padded to 32 at hash time).
    pub abstract_history: Vec<u8>,
```
with:
```rust
    /// 4-byte LE `history_signature()` of the hand so far — the trainer-exact
    /// hash input (see `compute_history_signature`).
    pub history_sig: u32,
```

### 2.2 Constructor
In `from_turn_state`, replace:
```rust
        let abstract_history = replay_abstract_history(config, &state.action_history)?;
```
with:
```rust
        let history_sig = compute_history_signature(&state.action_history);
```
and in the returned struct replace `abstract_history,` with `history_sig,`.
If the `config` parameter becomes unused, rename it `_config` (do not change the
signature shape otherwise).

### 2.3 Replace padding helper
Delete `padded_history()` (CZ L526–532) and add:
```rust
    /// The trainer's history slice: 4-byte LE history signature.
    pub fn history_bytes(&self) -> [u8; 4] {
        self.history_sig.to_le_bytes()
    }
```

### 2.4 Replace `replay_abstract_history` + delete `SeatMoney`
Delete the whole `SeatMoney` struct and the whole `replay_abstract_history` function
(CZ L539–654) and put this in their place:
```rust
/// Reproduce `pkr-core/src/state.rs::history_signature()` for the hand so far.
///
/// The current trainer hashes a 4-byte little-endian signature (NOT the old
/// abstract-bucket history):
///
/// ```text
/// bits 0..7   actions_this_street — fold/check/call/raise count in the
///             current phase (blinds/antes are NOT actions in the engine)
/// bits 8..15  raises — Bet ("raise") actions across the WHOLE hand so far
/// bit  16     last_was_bet — 1 iff the most recent counted action was a raise
/// ```
///
/// Mirrors `pkr-cfr/src/traversal.rs`:
/// `let sig = current.history_signature(); ... get_infoset_hash(hole, board, &sig.to_le_bytes(), street)`
///
/// Notes:
/// * blind/ante posts carry money but are not in the engine's action history
///   (`GameState::new` starts each hand with blinds already posted);
/// * a wire phase change corresponds to the engine's `advance_street_in_place`
///   resetting `actions_this_street` to 0;
/// * unknown wire actions are ignored (forward compat), exactly like before.
pub fn compute_history_signature(history: &[ActionEntry]) -> u32 {
    let mut actions_this_street: u32 = 0;
    let mut raises: u32 = 0;
    let mut last_was_bet: u32 = 0;
    let mut current_phase = String::new();

    for entry in history {
        if entry.phase != current_phase {
            current_phase = entry.phase.clone();
            actions_this_street = 0;
        }
        match entry.action.as_str() {
            "post_small_blind" | "post_big_blind" | "post_ante" => {}
            "fold" | "check" | "call" => {
                actions_this_street += 1;
                last_was_bet = 0;
            }
            "raise" => {
                actions_this_street += 1;
                raises += 1;
                last_was_bet = 1;
            }
            _ => {}
        }
    }

    (actions_this_street & 0xFF) | ((raises & 0xFF) << 8) | (last_was_bet << 16)
}
```

### 2.5 Replace `bucket_of_raise` (Change 3)
Replace the whole function (CZ L656–676) with:
```rust
/// Compute the abstract bucket of a *candidate* raise exactly the way the
/// CURRENT engine maps concrete actions to strategy slots —
/// `pkr-cfr/src/traversal.rs::abstract_action_index`, called with the
/// PRE-action state:
///
/// * all-in  : `total >= stack_before + sb_before`               → 5
/// * fraction: `total / max(pot, 1.2)` over the PRE-action pot
///   - `< 1.5` → 2
///   - `< 1.2` → 3   (DEAD branch in the engine too — kept verbatim)
///   - else    → 4
///
/// Do NOT "fix" the dead branch or reorder comparisons: byte-exact parity
/// with the trainer is the whole point.
pub fn bucket_of_raise(total: i64, stack_before: i64, sb_before: i64, pot: i64) -> u8 {
    let total_f = total as f32;
    if total_f >= (stack_before + sb_before) as f32 {
        return 5;
    }
    let pot_f = (pot as f32).max(1.2);
    let fraction = total_f / pot_f;
    if fraction < 1.5 {
        2
    } else if fraction < 1.2 {
        3
    } else {
        4
    }
}
```

### 2.6 Module doc header
In the module doc at the top of `src/adapter.rs`, replace point **1** (the paragraph
about the fixed `[u8; 32]` history, CZ L389–393) with:
```text
//! 1. **History** — the trainer hashes a 4-byte little-endian
//!    `history_signature()` (`pkr-core/src/state.rs`): actions this street
//!    (bits 0..7), total raises in the hand (bits 8..15), last-action-was-a-
//!    raise flag (bit 16). The bot reproduces it via
//!    [`compute_history_signature`].
```
and replace the whole "Abstract action buckets (K = 6)" table + the paragraph after it
(CZ L407–425) with:
```text
//! # Abstract action buckets (K = 6)
//!
//! From `pkr-cfr/src/traversal.rs::abstract_action_index` (PRE-action state):
//!
//! | bucket | meaning | condition |
//! |--------|---------|-----------|
//! | 0 | fold | `ActionKind::Fold` |
//! | 1 | check / call | `ActionKind::Check` or `ActionKind::Call` |
//! | 2 | bet/raise | `amount / max(pot_before, 1.2) < 1.5` |
//! | 3 | (dead) | unreachable in the engine — kept for parity |
//! | 4 | big bet/raise | `amount / max(pot_before, 1.2) >= 1.5`, not all-in |
//! | 5 | all-in | `amount >= stack_before + street_bet_before` |
//!
//! The trainer's legal bet fractions are `[0.5, 1.0, 2.0]` (+ all-in), capped at
//! 3 raises per street.
```

### 2.7 Replace adapter tests
Delete every test that calls `replay_abstract_history` and the old
`bucket_of_raise_matches_engine_semantics`, `golden_replay_of_documented_hand`,
`tiny_bet_buckets_to_two_and_big_raise_to_three`, `call_for_less_than_full_amount_still_buckets_one`,
`all_in_raise_buckets_to_five`, `ante_is_dead_money_not_street_bet`, and
`turn_context_pads_board_and_history` (CZ L707–872). Replace with:

```rust
    #[test]
    fn blinds_and_antes_are_not_actions() {
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ];
        assert_eq!(compute_history_signature(&history), 0);
    }

    #[test]
    fn signature_matches_engine_at_preflop_decision() {
        // seat0 raised to 30; we (seat1) are about to act.
        // engine state: actions_this_street=1, raises=1, last_was_bet=1.
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
        ];
        let sig = compute_history_signature(&history);
        assert_eq!(sig, 1 | (1 << 8) | (1 << 16));
    }

    #[test]
    fn signature_resets_per_street_and_counts_total_raises() {
        // Protocol §4 example hand. Decision point: flop, seat0 to act after
        // two flop checks: actions_this_street=2 (flop), raises=1 (preflop),
        // last_was_bet=0 (call closed preflop).
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
            entry_phase(0, "check", 0, "flop"),
        ];
        assert_eq!(compute_history_signature(&history), 2 | (1 << 8));
        // End of hand: flop actions=4, raises=2, last=fold.
        let full = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
            entry_phase(0, "check", 0, "flop"),
            entry_phase(0, "raise", 40, "flop"),
            entry_phase(1, "fold", 0, "flop"),
        ];
        assert_eq!(compute_history_signature(&full), 4 | (2 << 8));
    }

    #[test]
    fn bucket_of_raise_matches_new_engine_semantics() {
        // All-in: total >= stack_before + sb_before.
        assert_eq!(bucket_of_raise(200, 195, 5, 15), 5);
        // Opening 0.5x pot on pot 15: 10/15 = 0.33 -> 2.
        assert_eq!(bucket_of_raise(10, 995, 0, 15), 2);
        // Opening 1.0x pot: 20/15 = 1.33 -> 2  (OLD engine said 3!).
        assert_eq!(bucket_of_raise(20, 995, 0, 15), 2);
        // Opening 2.0x pot: 30/15 = 2.0 -> 4.
        assert_eq!(bucket_of_raise(30, 995, 0, 15), 4);
        // Pot-size raise facing a bet: raise to 100 on pot 80 -> 1.25 -> 2.
        assert_eq!(bucket_of_raise(100, 950, 0, 80), 2);
        // Big raise facing a bet: raise to 180 on pot 80 -> 2.25 -> 4.
        assert_eq!(bucket_of_raise(180, 950, 0, 80), 4);
    }

    #[test]
    fn turn_context_pads_board_and_carries_signature() {
        let state = TurnState {
            hand_number: 1,
            phase: "flop".into(),
            board: vec!["Ts".into(), "7h".into(), "2d".into()],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 80,
            your_stack: 970,
            opponent_stacks: vec![950],
            to_call: 20,
            min_raise: 40,
            max_raise: 970,
            action_history: vec![
                entry(0, "post_small_blind", 5),
                entry(1, "post_big_blind", 10),
                entry(0, "raise", 30),
                entry(1, "call", 30),
                entry_phase(1, "raise", 20, "flop"),
            ],
        };
        let ctx = TurnContext::from_turn_state(&state, &["fold".into()], &config()).unwrap();
        assert_eq!(ctx.hole, [25, 37]);
        assert_eq!(ctx.board, vec![8, 18, 26]);
        assert_eq!(ctx.street, 1);
        assert_eq!(ctx.padded_board(), [8, 18, 26, 0, 0]);
        // preflop: raise,call -> actions 2, raises 1, last=call(0); flop: raise so far.
        // actions_this_street=1, raises=2, last_was_bet=1.
        assert_eq!(ctx.history_sig, 1 | (2 << 8) | (1 << 16));
        assert_eq!(ctx.history_bytes(), (1u32 | (2 << 8) | (1 << 16)).to_le_bytes());
    }

    #[test]
    fn rejects_inconsistent_board_length() {
        // (keep this test exactly as it is in the file today — unchanged)
    }
```
Note: `config()`, `entry()`, `entry_phase()` helpers stay as they are.

Gate 2:
```bash
cargo check 2>&1 | head -50
```
Errors **inside `src/strategy.rs`** are expected until TASK 3 (it still references
`abstract_history`). There must be **zero errors attributed to `src/adapter.rs`**.
Do not run `cargo test` yet.

Commit: `T2: adapter — history_signature + new bucket_of_raise`.

---

## TASK 3 — `src/strategy.rs`: hash input, ladder, f32 compile fix

**You may touch**: `src/strategy.rs` only.

### 3.1 Compile fix (Change 1) — one line in `fallback_decision`
Replace (CZ L2711):
```rust
        let equity = calculate_ehs(&ctx.hole, &ctx.board, self.evaluator.as_ref()).0;
```
with:
```rust
        // calculate_ehs now returns (f32, f32) — widen once, all thresholds
        // below keep comparing against f64 literals.
        let equity = calculate_ehs(&ctx.hole, &ctx.board, self.evaluator.as_ref()).0 as f64;
```
(No other line of `fallback_decision` changes; `equity as f64` casts further down
remain valid.)

### 3.2 Hash input (Change 2) — `infoset_hash`
Replace the whole method (CZ L2672–2678):
```rust
    /// The trainer-exact infoset hash for this decision point.
    pub fn infoset_hash(&self, ctx: &TurnContext) -> u64 {
        let board = ctx.padded_board();
        let history = ctx.padded_history();
        self.abstraction
            .get_infoset_hash(&ctx.hole, &board, &history, ctx.street)
    }
```
with:
```rust
    /// The trainer-exact infoset hash for this decision point.
    ///
    /// Byte-exact mirror of `pkr-cfr/src/traversal.rs`:
    /// * board   = the trainer's fixed `[u8; 5]` (dealt cards + zero padding);
    /// * history = 4-byte LE `history_signature()` (see
    ///   `adapter::compute_history_signature`);
    /// * street  = engine street code 0..=3.
    pub fn infoset_hash(&self, ctx: &TurnContext) -> u64 {
        let board = ctx.padded_board();
        self.abstraction
            .get_infoset_hash(&ctx.hole, &board, &ctx.history_sig.to_le_bytes(), ctx.street)
    }
```

### 3.3 Ladder (Change 3) — `LADDER`
Replace (CZ L2584–2587):
```rust
pub const LADDER: [f64; 5] = [0.5, 0.75, 1.0, 1.5, 2.0];
```
with:
```rust
/// The trainer's bet-size ladder (`pkr-core/src/state.rs::legal_actions_into`):
/// raise totals are generated as `to_call + f * pot` (facing a bet) or
/// `f * pot` (opening), for these fractions in this order, plus an all-in
/// candidate handled separately. Current engine ladder: [0.5, 1.0, 2.0].
pub const LADDER: [f64; 3] = [0.5, 1.0, 2.0];
```

### 3.4 `ladder_sized_raise` doc comment
The function body works unchanged with the new `LADDER` and `bucket_of_raise`
(candidates → buckets → nearest-bucket match). Update only its doc comment: replace
the sentences mentioning the old five-fraction ladder and "sampled uniformly among
the ladder candidates" wording (CZ L2836–2845) with:
```rust
/// Generate concrete raise totals from the trainer's own size ladder
/// (`[0.5, 1.0, 2.0]` × pot + all-in) and pick one whose *engine bucket*
/// (see `adapter::bucket_of_raise`) equals the sampled slot.
///
/// This keeps runtime actions on-tree: the concrete sizes are exactly the
/// sizes the trainer's `legal_actions_into()` generated, so the chosen
/// (infoset, bucket) pair was actually explored during training. During
/// training the concrete size for a bucket was sampled uniformly among the
/// ladder candidates in that bucket — we mirror that: sampling mode draws a
/// uniform candidate among the best bucket-match set; `deterministic` mode
/// takes the first (smallest ladder size) for reproducibility.
```
Do not change the body. `map_slot`, `check_call`, `validate`, `pick_slot` and
`Engine::load` need **no changes**.

### 3.5 Strategy tests
1. In `ctx_stack` (CZ L2980–3001) replace `abstract_history: vec![1],` with
   `history_sig: 0,`.
2. Replace the test `ladder_sizing_lands_in_sampled_bucket` (CZ L3043–3067) with:
```rust
    #[test]
    fn ladder_sizing_lands_in_sampled_bucket() {
        // Opening bet, pot 80, stack 970 (max_raise = 970, my_sb = 0).
        // Ladder candidates: 40, 80, 160
        //   buckets (NEW engine math): 40/80=0.5 -> 2, 80/80=1.0 -> 2,
        //   160/80=2.0 -> 4. Slot 3 is dead in the engine.
        let c = ctx(&["check", "raise"], 0, 20, 970, 80);
        let d2 = ladder_sized_raise(&c, 2, true).expect("slot 2");
        assert_eq!(d2.amount.unwrap(), 40);
        assert_eq!(bucket_of_raise(40, c.your_stack, 0, c.pot), 2);
        let d4 = ladder_sized_raise(&c, 4, true).expect("slot 4");
        assert_eq!(d4.amount.unwrap(), 160);
        assert_eq!(bucket_of_raise(160, c.your_stack, 0, c.pot), 4);
        // Dead slot 3: nearest-bucket set is all three candidates
        // (|2-3|=1, |2-3|=1, |4-3|=1); deterministic mode takes the first.
        let d3 = ladder_sized_raise(&c, 3, true).expect("slot 3");
        assert_eq!(d3.amount.unwrap(), 40);
        // Sampling mode must stay inside the best set for the slot.
        for _ in 0..50 {
            let d = ladder_sized_raise(&c, 2, false).unwrap();
            assert_eq!(bucket_of_raise(d.amount.unwrap(), c.your_stack, 0, c.pot), 2);
        }
    }
```
3. `ladder_respects_min_raise_clamp` (CZ L3069–3075): keep as-is — with the new
   math the candidates are 20 (clamped), 30, 60 → buckets 2, 2, 4 → slot 2 takes 20,
   so the existing assertion still passes.
4. `ladder_avoids_accidental_shove_for_bet_slots`, `slot_five_shoves_max_raise`,
   `bet_slots_degrade_to_check_call_without_raise`, `validate_never_emits_illegal_actions`,
   and the CDF tests: keep as-is (verified consistent with the new engine).

Gate 3:
```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```
All three must pass. If clippy flags a dead-code/unused warning caused by this plan's
deletions (e.g. an import no longer used), remove that import — it is in scope.
Commit: `T3: strategy — 4-byte signature hash, new ladder, f32 equity`.

---

## TASK 4 — Touch-ups: main.rs comment + optional Cargo.toml cleanup

**You may touch**: `src/main.rs`, `Cargo.toml`.

### 4.1 Fix the stale `--ehs-samples` doc (CZ L1335–1339)
Replace:
```rust
    /// EHS Monte-Carlo samples per cluster computation. Use the SAME value
    /// your trainer used (run.sh exported EHS_SAMPLES=100; the engine's
    /// default is 1000). Overrides the EHS_SAMPLES env var.
```
with:
```rust
    /// EHS Monte-Carlo samples for the bot's own fallback heuristic
    /// (used only on blueprint misses). Does NOT affect infoset hashes — the
    /// current engine's hash path never runs EHS (the hash always takes the
    /// padded-5-board river arm). Overrides the EHS_SAMPLES env var.
```
Keep the flag and the `set_var` logic unchanged.

### 4.2 Remove the unused `pkr-export` dev-dependency (recommended)
Verified: no `pkr_export::` usage exists anywhere in the bot (grep of the whole
codebase). In `Cargo.toml` `[dev-dependencies]` delete:
```toml
# Test-only: build synthetic blueprint files (writer-side helpers).
pkr-export = { path = "../pkr-sota/crates/pkr-export" }
```
Then re-run Gate 3's three commands (must stay green).

Commit: `T4: touch-ups — stale ehs-samples doc, drop unused pkr-export dev-dep`.

---

## TASK 5 — Full build & test gate (whole bot)

```bash
cargo build --release
cargo test --release
```
Both must succeed. **Also confirm the pkr-sota crates compiled from the *current*
checkout**: `cargo tree -p pkr-chipzen-bot -i pkr-runtime` should list
`../pkr-sota/crates/pkr-runtime`. If any compile error points into `../pkr-sota/`,
STOP — the pkr-sota checkout is not the one this plan targets.

Commit: `T5: green build + tests against new pkr-sota`.

---

## TASK 6 — Load smoke test (no game server needed)

`runner.rs` loads the engine **before** connecting to the lobby (CZ L2399–2401), so a
deliberately unreachable lobby gives us a clean artifact-load check.

```bash
cargo run --release -- \
  --base-url ws://127.0.0.1:1 --bot-id smoke --token cz_extbot_smoke \
  --blueprint ../pkr-sota/outputs/smoke/blueprint.bin \
  --rank-table ../pkr-sota/outputs/smoke/hand_ranks.bin \
  --centroids ../pkr-sota/outputs/smoke/centroids.bin \
  --flop-buckets ../pkr-sota/outputs/smoke/flop_buckets.bin \
  --preflop-table ../pkr-sota/outputs/smoke/preflop_abstraction.bin \
  --flop-table ../pkr-sota/outputs/smoke/abstraction.bin \
  --turn-table ../pkr-sota/outputs/smoke/turn_abstraction.bin \
  --river-table ../pkr-sota/outputs/smoke/river_buckets.bin
```

**Pass** = log shows `blueprint loaded` with `version=2`, `max_actions_k=6`,
`infosets > 0`, followed by a (expected) lobby connection error.
**Fail** = `UnsupportedVersion(_)` or `InvalidHashAlgo` → the blueprint is old →
redo TASK 1. Any other artifact error → check the file paths.

Commit: `T6: load smoke against retrained blueprint`.

---

## TASK 7 — Live smoke (requires real credentials; optional here, mandatory before play)

Start the bot with `-vv` and the **same optional artifact flags the trainer used**
(from TASK 1's `run.sh` invocation: `--preflop-table`, `--flop-table`,
`--flop-buckets`, `--turn-table`, `--river-table`, `--rank-table`, `--centroids`).
The per-street `--*-centroids` flags are irrelevant to the hash; the ones that matter
are `--river-table` (board bucket in the hash) and `--flop-buckets` (flop byte in the
hash), but passing all of them guarantees trainer parity.

Watch the debug log over ≥ 20 decisions:
- `blueprint advice` (with `slot=` and `cdf=`) → **hit** (desired majority),
- `infoset miss → EHS fallback policy` → miss.

**Pass**: hits are the majority and raises look ladder-sized (0.5/1.0/2.0×pot or jam).
**Fail (100% misses)** → Appendix B, row 4.

---

## Appendix A — one-screen diff summary

| File | Change |
|------|--------|
| `src/adapter.rs` | `TurnContext.abstract_history: Vec<u8>` → `history_sig: u32`; `padded_history()` → `history_bytes()`; `replay_abstract_history` + `SeatMoney` deleted → `compute_history_signature`; `bucket_of_raise` → new engine math (pre-action pot, floor 1.2, threshold 1.5, dead 3); module doc updated; tests replaced |
| `src/strategy.rs` | `infoset_hash` passes `ctx.history_sig.to_le_bytes()`; `LADDER` `[0.5,0.75,1.0,1.5,2.0]` → `[0.5,1.0,2.0]`; `fallback_decision` equity cast `as f64`; doc comment of `ladder_sized_raise`; test `ctx_stack` field rename + ladder test expectations |
| `src/main.rs` | stale `--ehs-samples` doc text (flag kept) |
| `Cargo.toml` | drop unused `pkr-export` dev-dependency |
| artifacts | retrain via `../pkr-sota/run.sh` (old `blueprint.bin` rejected by V2 + hash-algo guard) |
| nothing else | runner/lobby/match_client/proto/error/config unchanged |

## Appendix B — failure signatures → cause → fix

| Symptom | Cause | Fix |
|---------|-------|-----|
| `error[E0308]: mismatched types` on `equity >= 0.62` in `strategy.rs` | TASK 3.1 skipped | apply the `as f64` cast |
| `MmapError::UnsupportedVersion(1)` at startup | old (V1) blueprint | TASK 1 retrain |
| `MmapError::InvalidHashAlgo(..)` at startup | blueprint built with legacy `DefaultHasher` | TASK 1 retrain |
| Blueprint loads, but ~100% `infoset miss` in play | hash inputs still differ from trainer | (a) TASK 2 signature math wrong — re-check bits/counts; (b) bot started without the trainer's optional artifacts — recheck TASK 7 flag list; (c) `hand_ranks.bin` differs from the one used in training — retrain + reuse the same outputs dir |
| Server rejects raises as illegal / sizes look odd | ladder/bucket mismatch | re-check TASK 2.5 `bucket_of_raise` and TASK 3.3 `LADDER` verbatim |
| `unresolved import pkr_abstraction::calculate_ehs` | wrong pkr-sota checkout | verify `../pkr-sota` is the current implementation repo |
| clippy dead-code after deletions | leftover import/const | delete the unused item (in scope) |

## Appendix C — API compatibility matrix (chipzen import → current pkr-sota)

| chipzen usage (file:symbol) | pkr-sota definition | Status |
|---|---|---|
| `pkr_abstraction::calculate_ehs` | `ehs.rs` — now `(f32, f32)` | ⚠ signature changed → TASK 3.1 |
| `pkr_abstraction::load_centroids` | `lib.rs` L1647 — `(&str) -> Result<CentroidStore, Box<dyn Error>>` | ✅ unchanged |
| `pkr_abstraction::KMeansAbstraction::{from_store, load_street_centroids, load_flop_buckets, init_table}` | `lib.rs` L1683–1721 | ✅ unchanged |
| `pkr_abstraction::KMeansAbstraction::get_infoset_hash` (via `AbstractionBuilder`) | `lib.rs` L1849 — format changed internally (length-prefixed history) | ✅ call-compatible; inputs must be the 4-byte signature (TASK 2/3) |
| `pkr_contracts::{AbstractionBuilder, SotaAdvice}` | `lib.rs` L5634–5650 | ✅ unchanged |
| `pkr_eval::TableEvaluator::new` (+`Evaluator` impl) | `lookup_fast.rs` L6575–6652 | ✅ unchanged |
| `pkr_runtime::{lookup::SolverHandle, MmapReader}` | `lookup.rs`/`mmap.rs` | ✅ unchanged surface; **loader now enforces V2+FNV** → TASK 1 |
| `MmapReader::file_header().{infoset_count, max_actions_k, version}` | `pkr-export/src/header.rs` L8004–8012 | ✅ unchanged |
| CDF semantics (`cdf_probabilities`, `(cum*255).round()`, K=6) | `pkr-export/src/writer.rs` L8383–8424 | ✅ unchanged → `pick_slot` untouched |
| trainer artifact load order vs `Engine::load` | trainer `main.rs` L527–557 | ✅ identical order |
| card encoding `suit*13 + rank`, engine street codes 0..=3 | `pkr-core` | ✅ unchanged (bot golden tests keep passing) |
