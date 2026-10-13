//! Adapter: chipzen Layer-2 `turn_request.state` → a live pkr-sota
//! `GameState` (trainer-exact by construction).
//!
//! # Blueprint compatibility contract
//!
//! The blueprint's infoset keys were produced by the *trainer*, which hashes
//! the engine's own `GameState` at every decision node
//! (`pkr-cfr/src/traversal.rs`):
//!
//! ```ignore
//! let mut sig_buf = [0u8; 8];
//! let sig_len = current.infoset_signature_into(&mut sig_buf);   // sig v1..v5 by env
//! let board = &current.board[..current.board_len as usize];     // valid slice
//! let hash = abstraction.get_infoset_hash(hole, board, &sig_buf[..sig_len], street);
//! ```
//!
//! Instead of re-implementing that byte math (three times already in this
//! repo's history — see `docs/rewire.md`), the adapter **replays the wire
//! action history through the engine's own `GameState`** and then calls the
//! very same engine functions. Any future engine change to signatures,
//! buckets, or street logic is inherited automatically; the loader's
//! abstraction fingerprint check rejects blueprints the replay cannot match.
//!
//! The replay mirrors the engine's own reference bot
//! (`pkr-runtime/src/bin/pkr-bot-pipe.rs`):
//!
//! 1. `GameState::new(stack_chips(), 1.0, 2.0)` — the engine's chip unit is
//!    *2 chips per big blind* at the trained depth (`PKR_STACK_BB`, default
//!    100). Wire chips are scaled by `2 / big_blind` into engine chips, so a
//!    chipzen 100bb match (stack 1000, blinds 5/10) replays as the engine's
//!    200-chip, 1/2 game.
//! 2. Blind posts are skipped (the engine's `new` starts with blinds
//!    already posted — identical money, no history entries).
//! 3. Every wire `fold/check/call/raise` maps to the engine action; a
//!    `raise` amount is a raise-to **total**, so it maps to
//!    `ActionKind::Bet(amount × scale)`.
//! 4. When the betting round closes mid-history the street is advanced with
//!    the next board cards (`is_street_complete` → `advance_street_in_place`),
//!    exactly like the pipe bot.
//! 5. Seat mapping: engine seat 0 is the small blind (the HU button). The
//!    wire's SB seat is read from `post_small_blind`; each history entry's
//!    seat must equal the wire seat of the engine's expected actor, which
//!    validates the replay against seat-order desyncs. Our own seat is the
//!    actor at the end of the replay — a `turn_request` is by definition
//!    *our* decision point.
//!
//! # Abstract action buckets (K = 7)
//!
//! `pkr-core/src/abstraction.rs::action_bucket` (PRE-action state):
//!
//! | bucket | meaning | condition |
//! |--------|---------|-----------|
//! | 0 | fold | `ActionKind::Fold` |
//! | 1 | check / call | `Check` or `Call` |
//! | 2 | small bet | raise-above-call / max(pot,1.2) < 0.75 |
//! | 3 | medium bet | < 1.5 |
//! | 4 | large bet | < 2.5 |
//! | 5 | overbet | >= 2.5 |
//! | 6 | jam | raise-to total >= stack + street bet |
//!
//! Sizing convention (§S7/B4): the fraction is **raise-above-call over the
//! pre-call pot** — `(amount − max(my_street_bet, opp_street_bet)) / pot`.
//! The trainer's legal sizings are `BET_SIZINGS = [0.5, 1.0, 2.0, 3.0]`
//! (+ all-in), capped at 3 raises per street; the strategy layer picks
//! among the engine's own `legal_actions_into()` candidates, so emitted
//! sizes are exactly on-tree.

use pkr_core::state::{stack_chips, Action, ActionKind, GameState};

use crate::error::{BotError, Result};
use crate::proto::cards::{parse_cards, phase_to_street};
use crate::proto::frames::{GameConfig, TurnState};

/// Number of abstract actions per infoset — single source of truth is the
/// engine itself (`NUM_ACTION_BUCKETS`, currently 7).
pub const K: usize = pkr_core::abstraction::NUM_ACTION_BUCKETS;

/// The jam bucket (`pkr-core::abstraction::JAM_BUCKET`).
pub const JAM_BUCKET: u8 = pkr_core::abstraction::JAM_BUCKET;

/// Wire → engine chip scale. The engine replays every 100 bb match as
/// `GameState::new(200, 1, 2)`: 2 chips per big blind.
pub fn wire_scale(big_blind: i64) -> f32 {
    2.0 / big_blind.max(1) as f32
}

/// One replayed step: an engine action, optionally followed by a street
/// advance with the given board cards. The session path (range tracker)
/// walks the same sequence the blueprint path applies.
#[derive(Debug, Clone)]
pub enum ReplayEvent {
    /// A poker action applied by the engine's current actor.
    Act(Action),
    /// The betting round closed; advance the street with these cards.
    Street(Vec<u8>),
}

/// Result of replaying a wire hand into the engine's coordinate system.
#[derive(Debug, Clone)]
pub struct Replay {
    /// Fresh deal with blinds posted and hole cards assigned (our real
    /// hole in `our_seat`, deterministic placeholders for the opponent).
    /// Feed this to `RuntimeSession::deal_start`.
    pub root: GameState,
    /// The full event sequence (actions + street advances).
    pub events: Vec<ReplayEvent>,
    /// The state at the decision node (after every history action).
    pub state: GameState,
    /// Our engine seat (0 = small blind / button).
    pub our_seat: u8,
    /// Wire → engine chip multiplier (`engine = wire × scale`).
    pub scale: f32,
    /// Wire seat that posted the small blind (engine seat 0).
    pub sb_wire_seat: i64,
}

/// Fully parsed decision context for one `turn_request` (wire-side data,
/// used for logging, the EHS fallback heuristic and wire-legality gates).
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub hand_number: i64,
    /// Engine street code (0..=3).
    pub street: u8,
    /// Our hole cards in engine encoding, `[card_a, card_b]`.
    pub hole: [u8; 2],
    /// Community cards actually dealt (engine encoding, 0/3/4/5 entries).
    pub board: Vec<u8>,
    /// Total pot including the current street's bets (wire chips).
    pub pot: i64,
    /// Our remaining stack (wire chips).
    pub your_stack: i64,
    /// Opponent stacks, wire order.
    pub opponent_stacks: Vec<i64>,
    /// Chips we must add to call; `0` = checking is free.
    pub to_call: i64,
    /// Minimum legal raise *total* (`0` when raising isn't legal).
    pub min_raise: i64,
    /// Maximum legal raise *total* (`0` when raising isn't legal).
    pub max_raise: i64,
    /// Legal action strings, wire order.
    pub valid_actions: Vec<String>,
    /// Match-level game config (blinds, stacks, ante).
    pub config: GameConfig,
    /// The raw wire state (the replay consumes this).
    pub state: TurnState,
}

impl TurnContext {
    /// Build the context from a `turn_request` payload + match config.
    pub fn from_turn_state(
        state: &TurnState,
        valid_actions: &[String],
        config: &GameConfig,
    ) -> Result<TurnContext> {
        if state.your_hole_cards.len() != 2 {
            return Err(BotError::Protocol(format!(
                "expected exactly 2 hole cards, got {}",
                state.your_hole_cards.len()
            )));
        }
        let hole_vec = parse_cards(&state.your_hole_cards)?;
        let hole = [hole_vec[0], hole_vec[1]];
        let board = parse_cards(&state.board)?;
        let street = phase_to_street(&state.phase)?;
        if board.len()
            != match street {
                0 => 0,
                1 => 3,
                2 => 4,
                _ => 5,
            }
        {
            return Err(BotError::Protocol(format!(
                "phase {:?} with {} board cards is inconsistent",
                state.phase,
                board.len()
            )));
        }
        Ok(TurnContext {
            hand_number: state.hand_number,
            street,
            hole,
            board,
            pot: state.pot,
            your_stack: state.your_stack,
            opponent_stacks: state.opponent_stacks.clone(),
            to_call: state.to_call,
            min_raise: state.min_raise,
            max_raise: state.max_raise,
            valid_actions: valid_actions.to_vec(),
            config: config.clone(),
            state: state.clone(),
        })
    }

    pub fn has_action(&self, action: &str) -> bool {
        self.valid_actions.iter().any(|a| a == action)
    }
}

/// Replay the wire hand into the engine's `GameState`.
///
/// Returns the full event trace plus the decision-node state. Errors on any
/// protocol/engine desync (bad alternation, board inconsistency, depth
/// mismatch) — the caller answers with a guaranteed-legal wire action.
pub fn replay_hand(state: &TurnState, config: &GameConfig) -> Result<Replay> {
    // ---- Match-level sanity ------------------------------------------------
    if config.variant != "nlhe" {
        return Err(BotError::Protocol(format!(
            "unsupported variant {:?} (blueprint is NLHE heads-up)",
            config.variant
        )));
    }
    if config.num_players != 2 {
        return Err(BotError::Protocol(format!(
            "heads-up blueprint; match has {} players",
            config.num_players
        )));
    }
    if config.ante != 0 {
        return Err(BotError::Protocol(
            "ante matches are unsupported (engine has no ante concept; \
             the shipped blueprint is ante-free)"
                .into(),
        ));
    }
    if config.big_blind <= 0 {
        return Err(BotError::Protocol("big_blind must be positive".into()));
    }
    let scale = wire_scale(config.big_blind);
    let engine_stack = stack_chips();
    if (config.starting_stack as f32 * scale - engine_stack).abs() > 1.0 {
        return Err(BotError::Protocol(format!(
            "match depth {} bb != trained depth {} bb — refusing to play a \
             mismatched blueprint (starting_stack={}, big_blind={})",
            config.starting_stack as f32 / config.big_blind as f32,
            engine_stack / 2.0,
            config.starting_stack,
            config.big_blind
        )));
    }

    // ---- Wire seat of the small blind (engine seat 0) ----------------------
    let sb_wire_seat = state
        .action_history
        .iter()
        .find(|e| e.action == "post_small_blind")
        .map(|e| e.seat)
        .unwrap_or(0);
    if sb_wire_seat != 0 && sb_wire_seat != 1 {
        return Err(BotError::Protocol(format!(
            "heads-up seats must be 0 and 1, got small blind on seat {sb_wire_seat}"
        )));
    }
    let seats = [sb_wire_seat, 1 - sb_wire_seat]; // wire seat of engine seat 0/1

    // ---- Build the root deal ----------------------------------------------
    let hole = parse_cards(&state.your_hole_cards)?;
    let board = parse_cards(&state.board)?;

    let mut root = GameState::new(stack_chips(), 1.0, 2.0);
    // Deterministic placeholder hole for the opponent that cannot collide
    // with our hole or the board (keeps engine debug-asserts happy).
    let mut used: Vec<u8> = hole.to_vec();
    used.extend_from_slice(&board);
    let mut villain = [0u8; 2];
    {
        let mut n = 0usize;
        for c in 0..52u8 {
            if n >= 2 {
                break;
            }
            if !used.contains(&c) {
                villain[n] = c;
                n += 1;
            }
        }
    }
    // Assign after constructing: engine seat 0 = SB. Fill both seats with
    // placeholders first, then overwrite ours once `our_seat` is known.
    root.hole[0] = [villain[0], villain[1]];
    root.hole[1] = [villain[0], villain[1]];

    // ---- Replay the action history ----------------------------------------
    let mut st = root.clone();
    let mut events: Vec<ReplayEvent> = Vec::with_capacity(state.action_history.len() + 3);
    let mut dealt = 0usize; // board cards consumed by street advances

    for entry in &state.action_history {
        match entry.action.as_str() {
            // The engine's `new` already posted blinds; antes are rejected.
            "post_small_blind" | "post_big_blind" | "post_ante" => continue,
            "fold" | "check" | "call" | "raise" => {}
            other => {
                tracing::warn!(action = other, "replay: ignoring unknown wire action");
                continue;
            }
        }
        // Seat validation: the engine applies to its own actor; the wire
        // entry must claim the same seat (mapped through the blinds).
        let expected_wire = seats[st.actor];
        if entry.seat != expected_wire {
            return Err(BotError::Protocol(format!(
                "replay desync: wire seat {} acted but engine expected seat {} \
                 (actor {}); hand history does not alternate like a legal HU hand",
                entry.seat, expected_wire, st.actor
            )));
        }
        let kind = match entry.action.as_str() {
            "fold" => ActionKind::Fold,
            "check" => ActionKind::Check,
            "call" => ActionKind::Call,
            _ => {
                if entry.amount <= 0 {
                    return Err(BotError::Protocol(format!(
                        "raise with non-positive amount {}",
                        entry.amount
                    )));
                }
                ActionKind::Bet(entry.amount as f32 * scale)
            }
        };
        let act = Action {
            player: st.actor,
            kind,
        };
        st.apply_action_in_place(&act);
        events.push(ReplayEvent::Act(act));
        if st.is_terminal() {
            // A fold ended the hand; a turn_request cannot legally follow.
            return Err(BotError::Protocol(
                "replay desync: hand reached a terminal state before the \
                 decision point"
                    .into(),
            ));
        }
        if st.is_street_complete() && st.street != pkr_core::state::Street::River {
            let need = match st.street {
                pkr_core::state::Street::Preflop => 3,
                pkr_core::state::Street::Flop => 1,
                _ => 1,
            };
            if dealt + need > board.len() {
                return Err(BotError::Protocol(format!(
                    "replay desync: street closed but the wire board has only \
                     {} cards (need {} more)",
                    board.len(),
                    need
                )));
            }
            let chunk = board[dealt..dealt + need].to_vec();
            dealt += need;
            st.advance_street_in_place(&chunk);
            events.push(ReplayEvent::Street(chunk));
        }
    }

    // ---- Validate the replay against the wire snapshot ---------------------
    let wire_street = phase_to_street(&state.phase)?;
    if st.street as u8 != wire_street {
        return Err(BotError::Protocol(format!(
            "replay desync: engine street {} != wire phase {} ({:?})",
            st.street as u8,
            wire_street,
            st.street
        )));
    }
    if st.board_len as usize != board.len() {
        return Err(BotError::Protocol(format!(
            "replay desync: engine board_len {} != wire board len {}",
            st.board_len, board.len()
        )));
    }
    if st.is_terminal() {
        return Err(BotError::Protocol(
            "decision requested on a terminal state".into(),
        ));
    }

    // A `turn_request` is by definition our decision point: the engine's
    // current actor is us.
    let our_seat = st.actor as u8;
    root.hole[our_seat as usize] = [hole[0], hole[1]];
    st.hole[our_seat as usize] = [hole[0], hole[1]];

    Ok(Replay {
        root,
        events,
        state: st,
        our_seat,
        scale,
        sb_wire_seat,
    })
}

/// The engine's canonical infoset signature bytes for a decision state
/// (dispatches on the `PKR_SIG_*` env — sig 5 for the shipped k7-baseline).
pub fn infoset_signature(st: &GameState) -> Vec<u8> {
    let mut buf = [0u8; 8];
    let len = st.infoset_signature_into(&mut buf);
    buf[..len].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::frames::ActionEntry;

    fn config() -> GameConfig {
        GameConfig {
            variant: "nlhe".into(),
            starting_stack: 1000,
            small_blind: 5,
            big_blind: 10,
            ante: 0,
            num_players: 2,
        }
    }

    fn entry(seat: i64, action: &str, amount: i64) -> ActionEntry {
        entry_phase(seat, action, amount, "preflop")
    }

    fn entry_phase(seat: i64, action: &str, amount: i64, phase: &str) -> ActionEntry {
        ActionEntry {
            seat,
            action: action.into(),
            amount,
            phase: phase.into(),
            is_timeout: false,
        }
    }

    fn turn(state: &TurnState) -> TurnContext {
        TurnContext::from_turn_state(state, &["check".to_string()], &config()).unwrap()
    }

    /// Protocol §4 example hand. SB raises to 30, BB calls; on the flop BB
    /// checks — the decision point is SB (wire seat 0) acting free. The
    /// engine must replay it cleanly with the flop actor decided by the
    /// engine's own alternation.
    #[test]
    fn replays_documented_flop_hand() {
        let s = TurnState {
            hand_number: 1,
            phase: "flop".into(),
            board: vec!["Ts".into(), "7h".into(), "2d".into()],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 60,
            your_stack: 970,
            opponent_stacks: vec![970],
            to_call: 0,
            min_raise: 20,
            max_raise: 970,
            action_history: vec![
                entry(0, "post_small_blind", 5),
                entry(1, "post_big_blind", 10),
                entry(0, "raise", 30),
                entry(1, "call", 30),
                entry_phase(1, "check", 0, "flop"),
            ],
        };
        let ctx = turn(&s);
        let rep = replay_hand(&ctx.state, &ctx.config).expect("replay");
        // Wire seat 0 posted the SB → engine seat 0. The last action was
        // BB's flop check, so the decision actor is the SB (wire seat 0,
        // engine seat 0).
        assert_eq!(rep.sb_wire_seat, 0);
        assert_eq!(rep.our_seat, 0);
        assert_eq!(rep.scale, 0.2);
        assert_eq!(rep.state.street as u8, 1);
        assert_eq!(rep.state.board_len, 3);
        assert_eq!(rep.state.actor as u8, rep.our_seat);
        // Money (engine chips): blinds 1+2 in the pot; raise-to-30 wire =
        // 6 engine (adds 6−1 = 5); call adds 6−2 = 4. Pot = 3+5+4 = 12
        // (= 60 wire × 0.2); street bets reset to 0 at the flop advance.
        assert_eq!(rep.state.pot, 12.0);
        assert_eq!(rep.state.street_bets, [0.0, 0.0]);
        assert_eq!(rep.state.stacks, [200.0 - 6.0, 200.0 - 6.0]);
        // Events: raise, call, street advance, check.
        assert_eq!(rep.events.len(), 4);
        assert!(matches!(rep.events[2], ReplayEvent::Street(_)));
    }

    /// Button on wire seat 1: seat 1 posts the SB and is engine seat 0.
    /// With only blinds posted, the preflop actor is the SB — and the
    /// decision context (to_call = 5, stack 995 = 1000 − SB) belongs to
    /// the SB, so our engine seat is 0.
    #[test]
    fn maps_seats_through_the_button() {
        let s = TurnState {
            hand_number: 2,
            phase: "preflop".into(),
            board: vec![],
            your_hole_cards: vec!["Qs".into(), "Qh".into()],
            pot: 15,
            your_stack: 995,
            opponent_stacks: vec![1000],
            to_call: 5,
            min_raise: 10,
            max_raise: 995,
            action_history: vec![
                entry(1, "post_small_blind", 5),
                entry(0, "post_big_blind", 10),
            ],
        };
        let ctx = turn(&s);
        let rep = replay_hand(&ctx.state, &ctx.config).expect("replay");
        assert_eq!(rep.sb_wire_seat, 1);
        // The preflop actor is the SB (wire seat 1) → engine seat 0 = us.
        assert_eq!(rep.our_seat, 0);
        assert_eq!(rep.state.actor as u8, 0);
        assert_eq!(rep.state.street_bets, [1.0, 2.0]);
    }

    /// A raise to 30 wire chips is a raise to 6 engine chips (scale 0.2).
    #[test]
    fn scales_wire_chips_into_engine_chips() {
        let s = TurnState {
            hand_number: 1,
            phase: "preflop".into(),
            board: vec![],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 40, // 30 (SB raise-to) + 10 (BB)
            your_stack: 970,
            opponent_stacks: vec![975],
            to_call: 25,
            min_raise: 30,
            max_raise: 970,
            action_history: vec![
                entry(0, "post_small_blind", 5),
                entry(1, "post_big_blind", 10),
                entry(0, "raise", 30),
            ],
        };
        let ctx = turn(&s);
        let rep = replay_hand(&ctx.state, &ctx.config).expect("replay");
        assert_eq!(rep.our_seat, 1); // we are BB, facing the raise
        assert_eq!(rep.state.street_bets, [6.0, 2.0]); // 6 = 30 × 0.2
        assert_eq!(rep.state.pot, 8.0); // blinds 3 + raise adds 5 (= 40 wire × 0.2)
    }

    /// Seat-order desyncs must be rejected, not silently replayed.
    #[test]
    fn rejects_bad_seat_alternation() {
        let s = TurnState {
            hand_number: 1,
            phase: "preflop".into(),
            board: vec![],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 45,
            your_stack: 960,
            opponent_stacks: vec![970],
            to_call: 0,
            min_raise: 20,
            max_raise: 960,
            action_history: vec![
                entry(0, "post_small_blind", 5),
                entry(1, "post_big_blind", 10),
                // SB raising twice in a row is impossible in HU.
                entry(0, "raise", 30),
                entry(0, "raise", 60),
            ],
        };
        let ctx = turn(&s);
        assert!(replay_hand(&ctx.state, &ctx.config).is_err());
    }

    /// Non-100bb matches are refused — the fingerprint pins the depth.
    #[test]
    fn rejects_depth_mismatch() {
        let cfg = GameConfig {
            variant: "nlhe".into(),
            starting_stack: 500,
            small_blind: 5,
            big_blind: 10, // 50 bb — blueprint is 100 bb
            ante: 0,
            num_players: 2,
        };
        let s = TurnState {
            hand_number: 1,
            phase: "preflop".into(),
            board: vec![],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 15,
            your_stack: 495,
            opponent_stacks: vec![500],
            to_call: 5,
            min_raise: 10,
            max_raise: 495,
            action_history: vec![
                entry(0, "post_small_blind", 5),
                entry(1, "post_big_blind", 10),
            ],
        };
        let ctx = TurnContext::from_turn_state(&s, &["call".to_string()], &cfg).unwrap();
        let err = replay_hand(&ctx.state, &ctx.config).unwrap_err();
        assert!(err.to_string().contains("depth"), "got: {err}");
    }

    /// The engine's K=7 bucket mapping, exercised through a replayed raise:
    /// a 1.0×-pot raise lands in bucket 3 (raise-above-call / pot = 1.0).
    #[test]
    fn k7_bucket_semantics_via_engine() {
        // Preflop: SB 1, BB 2 (engine chips). BB opened to 6 (opp=6), pot=8.
        // Raising to 14 total is a raise OF 8 = 1.0× the pre-call pot.
        let mut st = GameState::new(200.0, 1.0, 2.0);
        st.apply_action_in_place(&Action {
            player: st.actor,
            kind: ActionKind::Bet(6.0),
        });
        let b = pkr_core::abstraction::action_bucket(
            &ActionKind::Bet(14.0),
            st.stacks[st.actor],
            st.street_bets[st.actor],
            st.street_bets[1 - st.actor],
            st.pot,
        );
        assert_eq!(b, 3, "raise-above-call 8/8 = 1.0 must be bucket 3");
        // 0.5× pot raise → bucket 2; 2.0× → 4; 3.0× → 5; jam → 6.
        assert_eq!(
            pkr_core::abstraction::action_bucket(
                &ActionKind::Bet(10.0),
                st.stacks[st.actor],
                st.street_bets[st.actor],
                st.street_bets[1 - st.actor],
                st.pot
            ),
            2
        );
        assert_eq!(
            pkr_core::abstraction::action_bucket(
                &ActionKind::Bet(22.0),
                st.stacks[st.actor],
                st.street_bets[st.actor],
                st.street_bets[1 - st.actor],
                st.pot
            ),
            4
        );
        assert_eq!(
            pkr_core::abstraction::action_bucket(
                &ActionKind::Bet(30.0),
                st.stacks[st.actor],
                st.street_bets[st.actor],
                st.street_bets[1 - st.actor],
                st.pot
            ),
            5
        );
        assert_eq!(
            pkr_core::abstraction::action_bucket(
                &ActionKind::Bet(200.0),
                st.stacks[st.actor],
                st.street_bets[st.actor],
                st.street_bets[1 - st.actor],
                st.pot
            ),
            pkr_core::abstraction::JAM_BUCKET
        );
    }
}
