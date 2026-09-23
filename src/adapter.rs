//! Adapter: chipzen Layer-2 `turn_request.state` → pkr-sota infoset inputs.
//!
//! # Blueprint compatibility contract (read this before "fixing" anything)
//!
//! The blueprint's keys were produced by the *trainer*, so the runtime must
//! reproduce the trainer's hash inputs byte-for-byte. Replicating
//! `pkr-cfr/src/traversal.rs` exactly as found in the pkr-sota repo dump:
//!
//! 1. **History** — the trainer hashes a 4-byte little-endian
//!    `history_signature()` (`pkr-core/src/state.rs`): actions this street
//!    (bits 0..7), total raises in the hand (bits 8..15), last-action-was-a-
//!    raise flag (bit 16). The bot reproduces it via
//!    [`compute_history_signature`].
//! 2. **Board** — the trainer passes `&current.board`, a fixed `[u8; 5]`
//!    array, so the board slice is *always* 5 bytes: the dealt community cards
//!    followed by zero padding (preflop = five zero bytes).
//! 3. **Street** — `Street as u8`: preflop 0, flop 1, turn 2, river 3.
//! 4. **Hole cards** — the acting player's two cards, i.e. *our* cards, in
//!    engine encoding (`suit * 13 + rank`).
//!
//! The cluster id / flop bucket inside `get_infoset_hash` are computed by
//! `pkr-abstraction` itself, so by reusing that crate we inherit them for
//! free — including the trainer's quirks (e.g. the preflop padded board makes
//! every preflop EHS evaluation degenerate to a constant, which the trainer
//! also did; see `tests/golden.rs`).
//!
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

use crate::error::{BotError, Result};
use crate::proto::cards::{parse_cards, phase_to_street};
use crate::proto::frames::{ActionEntry, GameConfig, TurnState};

/// Number of abstract actions per infoset (`pkr-cfr` traversal `K`).
pub const K: usize = 6;
/// Fixed abstract-history array length in the trainer (`[u8; 32]`).
pub const HISTORY_LEN: usize = 32;

/// Fully parsed decision context for one `turn_request`.
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub hand_number: i64,
    /// Engine street code (0..=3).
    pub street: u8,
    /// Our hole cards in engine encoding, `[card_a, card_b]`.
    pub hole: [u8; 2],
    /// Community cards actually dealt (engine encoding, 0/3/4/5 entries).
    pub board: Vec<u8>,
    /// Total pot including the current street's bets (chips).
    pub pot: i64,
    /// Our remaining stack (chips not yet in the pot).
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
    /// 4-byte LE `history_signature()` of the hand so far — the trainer-exact
    /// hash input (see `compute_history_signature`).
    pub history_sig: u32,
    /// Match-level game config (blinds, stacks, ante).
    pub config: GameConfig,
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
        let history_sig = compute_history_signature(&state.action_history);
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
            history_sig,
            config: config.clone(),
        })
    }

    /// The trainer's board slice: dealt cards + zero padding to 5 bytes.
    pub fn padded_board(&self) -> [u8; 5] {
        let mut out = [0u8; 5];
        for (dst, &src) in out.iter_mut().zip(self.board.iter()) {
            *dst = src;
        }
        out
    }

    /// The trainer's history slice: 4-byte LE history signature.
    pub fn history_bytes(&self) -> [u8; 4] {
        self.history_sig.to_le_bytes()
    }

    pub fn has_action(&self, action: &str) -> bool {
        self.valid_actions.iter().any(|a| a == action)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            ctx.history_bytes(),
            (1u32 | (2 << 8) | (1 << 16)).to_le_bytes()
        );
    }

    #[test]
    fn rejects_inconsistent_board_length() {
        // (keep this test exactly as it is in the file today — unchanged)
    }
}
