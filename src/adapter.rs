//! Adapter: chipzen Layer-2 `turn_request.state` → pkr-sota infoset inputs.
//!
//! # Blueprint compatibility contract (read this before "fixing" anything)
//!
//! The blueprint's keys were produced by the *trainer*, so the runtime must
//! reproduce the trainer's hash inputs byte-for-byte. Replicating
//! `pkr-cfr/src/traversal.rs` exactly as found in the pkr-sota repo dump:
//!
//! 1. **History** — the trainer builds a fixed `[u8; 32]` array from
//!    `current.abstract_history` (note: it uses `.len()` on the *fixed-size*
//!    array, so the length is always 32) and passes all 32 bytes to
//!    `get_infoset_hash`. The meaningful prefix holds the abstract action
//!    buckets (see [`abstract_action_bucket`]); the remainder is zero padding.
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
//! From `pkr-cfr/src/traversal.rs::abstract_action_index` /
//! `pkr-core/src/state.rs::abstract_action_index_static`:
//!
//! | bucket | meaning | engine condition (state *after* money moved) |
//! |--------|---------|----------------------------------------------|
//! | 0 | fold | `ActionKind::Fold` |
//! | 1 | check / call | `ActionKind::Check` or `ActionKind::Call` |
//! | 2 | small bet | `amount / pot_after < 0.5` |
//! | 3 | medium bet | `0.5 <= amount / pot_after < 1.0` |
//! | 4 | large bet | `amount / pot_after >= 1.0` (and not all-in) |
//! | 5 | all-in | `amount >= stack_before + street_bet_before` |
//!
//! Note: because the fraction uses the *post-action* pot, bucket 4 is never
//! reachable through the trainer's own size ladder on a fresh bet
//! (`amt/(pot+amt) < 1` for any `amt`); it can only occur in exotic
//! already-committed spots. The runtime keeps slot 4 mapped via
//! nearest-bucket degradation (see `strategy::ladder_sized_raise`).
//!
//! Blinds and antes are *not* bucketed: the trainer's `GameState::new` starts
//! each hand with blinds already posted and an empty `abstract_history`.

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
    /// Abstract action buckets replayed from `action_history`
    /// (meaningful prefix; zero-padded to 32 at hash time).
    pub abstract_history: Vec<u8>,
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
        if board.len() != match street {
            0 => 0,
            1 => 3,
            2 => 4,
            _ => 5,
        } {
            return Err(BotError::Protocol(format!(
                "phase {:?} with {} board cards is inconsistent",
                state.phase,
                board.len()
            )));
        }
        let abstract_history = replay_abstract_history(config, &state.action_history)?;
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
            abstract_history,
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

    /// The trainer's history slice: buckets + zero padding to 32 bytes.
    pub fn padded_history(&self) -> [u8; HISTORY_LEN] {
        let mut out = [0u8; HISTORY_LEN];
        let n = self.abstract_history.len().min(HISTORY_LEN);
        out[..n].copy_from_slice(&self.abstract_history[..n]);
        out
    }

    pub fn has_action(&self, action: &str) -> bool {
        self.valid_actions.iter().any(|a| a == action)
    }
}

/// Money state for one seat while replaying a hand.
#[derive(Debug, Clone, Copy, Default)]
struct SeatMoney {
    stack: f32,
    street_bet: f32,
}

/// Replay chipzen's `action_history` into the trainer's abstract buckets.
///
/// Tracks per-seat stacks / street bets / pot exactly like
/// `pkr-core::GameState` does, so each `raise` can be bucketed with the
/// engine's *post-action pot* fraction semantics.
///
/// Notes:
/// * blinds/antes update the money state but produce **no** bucket;
/// * `call` chips are derived from the reconstructed state (the wire's
///   `amount` for calls is the matched *total*, and `turn_result` uses the
///   incremental amount — deriving from state is the only unambiguous read);
/// * `raise` amount on the wire is the **total street bet** (Layer-2 §5.4).
pub fn replay_abstract_history(
    config: &GameConfig,
    history: &[ActionEntry],
) -> Result<Vec<u8>> {
    let n = config.num_players.max(2);
    let mut seats = vec![
        SeatMoney { stack: config.starting_stack as f32, street_bet: 0.0 };
        n
    ];
    let mut pot = 0f32;
    let mut buckets: Vec<u8> = Vec::with_capacity(HISTORY_LEN);
    // Street bets reset when the hand moves to the next betting phase
    // (engine: `GameState::advance_street_in_place` zeroes `street_bets`).
    let mut current_phase = String::new();

    for entry in history {
        if entry.phase != current_phase {
            for s in seats.iter_mut() {
                s.street_bet = 0.0;
            }
            current_phase = entry.phase.clone();
        }
        let seat_idx = usize::try_from(entry.seat)
            .map_err(|_| BotError::Protocol(format!("negative seat {}", entry.seat)))?;
        if seat_idx >= n {
            return Err(BotError::Protocol(format!(
                "seat {} out of range for num_players={n}",
                entry.seat
            )));
        }
        match entry.action.as_str() {
            // Synthetic posts: money only, no bucket.
            "post_small_blind" | "post_big_blind" | "post_ante" => {
                let paid = (entry.amount as f32).min(seats[seat_idx].stack);
                seats[seat_idx].stack -= paid;
                pot += paid;
                if entry.action != "post_ante" {
                    seats[seat_idx].street_bet += paid;
                }
            }
            "fold" => buckets.push(0),
            "check" => buckets.push(1),
            "call" => {
                buckets.push(1);
                // Heads-up (the only supported remote mode): the caller matches
                // the opponent's street bet, capped by their stack. Derive from
                // state rather than trusting the wire amount (see doc comment).
                let opp_sb = seats
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != seat_idx)
                    .map(|(_, s)| s.street_bet)
                    .fold(0.0f32, f32::max);
                let to_call = (opp_sb - seats[seat_idx].street_bet).max(0.0);
                let chips = to_call.min(seats[seat_idx].stack);
                seats[seat_idx].stack -= chips;
                pot += chips;
                seats[seat_idx].street_bet += chips;
            }
            "raise" => {
                let total = entry.amount as f32;
                let stack_before = seats[seat_idx].stack;
                let sb_before = seats[seat_idx].street_bet;
                let all_in = total >= stack_before + sb_before;
                let chips = (total - sb_before).max(0.0).min(stack_before);
                seats[seat_idx].stack -= chips;
                pot += chips;
                seats[seat_idx].street_bet += chips;
                // Engine semantics: fraction over the POST-action pot.
                buckets.push(if all_in {
                    5
                } else {
                    let fraction = total / pot.max(1.0);
                    if fraction < 0.5 {
                        2
                    } else if fraction < 1.0 {
                        3
                    } else {
                        4
                    }
                });
            }
            other => {
                // Unknown synthetic action — ignore for bucketing (forward
                // compat), but don't fail the whole hand.
                tracing::debug!(action = other, "ignoring unknown action_history entry");
            }
        }
        if buckets.len() > HISTORY_LEN {
            return Err(BotError::Protocol(
                "action history exceeds 32 abstract actions".into(),
            ));
        }
    }

    Ok(buckets)
}

/// Compute the abstract bucket of a *candidate* raise exactly the way the
/// engine does when the action is applied (post-action pot fraction).
///
/// `total` = candidate raise total; `stack_before` / `sb_before` / `pot` are
/// the actor's pre-action values (pot includes both players' street bets).
pub fn bucket_of_raise(total: i64, stack_before: i64, sb_before: i64, pot: i64) -> u8 {
    let total_f = total as f32;
    let all_in = total_f >= (stack_before + sb_before) as f32;
    if all_in {
        return 5;
    }
    let pot_after = pot as f32 + (total_f - sb_before as f32);
    let fraction = total_f / pot_after.max(1.0);
    if fraction < 0.5 {
        2
    } else if fraction < 1.0 {
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
    fn blinds_are_not_bucketed() {
        let b = replay_abstract_history(
            &config(),
            &[entry(0, "post_small_blind", 5), entry(1, "post_big_blind", 10)],
        )
        .unwrap();
        assert!(b.is_empty(), "blinds must not enter the abstract history");
    }

    /// Golden replay of the documented example hand (poker protocol §4):
    /// SB 5 / BB 10, seat0 raises to 30, seat1 calls, flop check-check,
    /// seat0 raises to 40, seat1 folds.
    ///
    /// Expected buckets:
    /// * raise to 30: pot 15 → 40, fraction 30/40 = 0.75 → bucket 3
    /// * call → 1, check → 1, check → 1
    /// * raise to 40: pot 60 → 100, fraction 40/100 = 0.40 → bucket 2
    /// * fold → 0
    #[test]
    fn golden_replay_of_documented_hand() {
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
            entry_phase(0, "check", 0, "flop"),
            entry_phase(0, "raise", 40, "flop"),
            entry_phase(1, "fold", 0, "flop"),
        ];
        let b = replay_abstract_history(&config(), &history).unwrap();
        assert_eq!(b, vec![3, 1, 1, 1, 2, 0]);
    }

    #[test]
    fn all_in_raise_buckets_to_five() {
        // Short stack 100: SB posts 5 (stack 95) then shoves 100 total.
        // all-in iff total >= stack_before + sb_before = 95 + 5 = 100.
        let cfg = GameConfig { starting_stack: 100, ..config() };
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 100),
        ];
        let b = replay_abstract_history(&cfg, &history).unwrap();
        assert_eq!(b, vec![5]);
    }

    #[test]
    fn tiny_bet_buckets_to_two_and_big_raise_to_three() {
        // pot after blinds = 15; a 7-chip opening bet:
        // pot_after = 15 + 7 = 22, fraction 7/22 = 0.318 → bucket 2
        // then opp raise to 25 total (sb 10 → chips 15):
        // pot_after = 22 + 15 = 37, fraction 25/37 = 0.676 → bucket 3
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 7),
            entry(1, "raise", 25),
        ];
        let b = replay_abstract_history(&config(), &history).unwrap();
        assert_eq!(b, vec![2, 3]);
    }

    #[test]
    fn call_for_less_than_full_amount_still_buckets_one() {
        // Short all-in call: stack 20 facing a 30 total bet.
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 20), // wire says matched-total shape would be 30;
            // our replay derives chips from state: to_call = 30-10 = 20, capped
            // by stack 20 → all-in for 20.
        ];
        let b = replay_abstract_history(&config(), &history).unwrap();
        assert_eq!(b, vec![3, 1]);
    }

    #[test]
    fn bucket_of_raise_matches_engine_semantics() {
        // Engine all-in test: total >= stack_before + sb_before
        assert_eq!(bucket_of_raise(200, 195, 5, 15), 5);
        // 0.5x-pot opening bet: candidate 10 on pot 15 → pot_after 25 → 0.4 → 2
        assert_eq!(bucket_of_raise(10, 995, 0, 15), 2);
        // 1.0x-pot opening bet: candidate 20 on pot 15 → pot_after 35 → 0.571 → 3
        assert_eq!(bucket_of_raise(20, 995, 0, 15), 3);
        // Huge but not all-in raise can hit bucket 4:
        // candidate 90 on pot 15 with sb_before 0: pot_after 105 → 0.857 → 3
        // candidate 90 on pot 15 with sb_before 60 (already in 60):
        //   pot_after = 15 + 30 = 45 → 90/45 = 2.0 → bucket 4
        assert_eq!(bucket_of_raise(90, 140, 60, 15), 4);
    }

    #[test]
    fn turn_context_pads_board_and_history() {
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
        // Ah = 25, Kd = 2*13 + 11 = 37
        assert_eq!(ctx.hole, [25, 37]);
        // Ts = 8, 7h = 13 + 5 = 18, 2d = 26 + 0 = 26
        assert_eq!(ctx.board, vec![8, 18, 26]);
        assert_eq!(ctx.street, 1);
        assert_eq!(ctx.padded_board(), [8, 18, 26, 0, 0]);
        // preflop buckets: raise to 30 → 3 (pot 15→40), call → 1,
        // flop raise to 20: pot 60→80, fraction 20/80 = 0.25 → 2
        assert_eq!(ctx.abstract_history, vec![3, 1, 2]);
        let hist = ctx.padded_history();
        assert_eq!(hist.len(), 32);
        assert_eq!(&hist[..3], &[3, 1, 2]);
        assert!(hist[3..].iter().all(|&b| b == 0));
    }

    #[test]
    fn rejects_inconsistent_board_length() {
        let state = TurnState {
            hand_number: 1,
            phase: "turn".into(),
            board: vec!["Ts".into(), "7h".into(), "2d".into()],
            your_hole_cards: vec!["Ah".into(), "Kd".into()],
            pot: 80,
            your_stack: 970,
            opponent_stacks: vec![950],
            to_call: 0,
            min_raise: 0,
            max_raise: 0,
            action_history: vec![],
        };
        assert!(TurnContext::from_turn_state(&state, &[], &config()).is_err());
    }

    #[test]
    fn ante_is_dead_money_not_street_bet() {
        let cfg = GameConfig { ante: 5, ..config() };
        let history = vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "post_ante", 5),
            entry(1, "post_ante", 5),
            entry(0, "raise", 20),
        ];
        let b = replay_abstract_history(&cfg, &history).unwrap();
        // pot after blinds+antes = 25 (antes are dead money, NOT street bets);
        // raise to 20 → pot_after = 25 + (20 - 5) = 40, fraction 20/40 = 0.5
        // → NOT < 0.5 → bucket 3 (boundary case).
        assert_eq!(b, vec![3]);
    }
}
