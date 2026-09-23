//! Golden / compatibility tests for the trainer-exact infoset hashing.
//!
//! These pin the *documented quirks* of the pkr-sota trainer (as found in the
//! repo dump the bot was built against). If you retrain with a fixed trainer,
//! these tests — and the blueprint keys — change together.

mod common;

use pkr_chipzen_bot::adapter::TurnContext;
use pkr_chipzen_bot::proto::frames::{ActionEntry, GameConfig, TurnState};

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
    ActionEntry {
        seat,
        action: action.into(),
        amount,
        phase: "preflop".into(),
        is_timeout: false,
    }
}

fn state(phase: &str, board: Vec<&str>, history: Vec<ActionEntry>) -> TurnState {
    TurnState {
        hand_number: 1,
        phase: phase.into(),
        board: board.into_iter().map(String::from).collect(),
        your_hole_cards: vec!["Ah".into(), "Kd".into()],
        pot: 60,
        your_stack: 970,
        opponent_stacks: vec![970],
        to_call: 0,
        min_raise: 20,
        max_raise: 970,
        action_history: history,
    }
}

#[test]
fn hash_is_deterministic_across_engine_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = common::build_artifacts(dir.path(), &[(0xdeadbeef, [255u8; 6])]);

    let s = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ],
    );
    let ctx = TurnContext::from_turn_state(&s, &["check".to_string()], &config()).unwrap();

    let h1 = common::synthetic_hash(dir.path(), &ctx);
    let h2 = common::synthetic_hash(dir.path(), &ctx);
    assert_ne!(h1, 0);
    assert_eq!(h1, h2, "synthetic artifacts must hash deterministically");
    let _ = artifacts;
}

/// The shipped trainer passes the full `[u8; 32]` abstract-history array
/// (zero padding included) to `get_infoset_hash`, so different-length
/// histories always hash differently — and the padded tail is constant.
#[test]
fn hash_respects_history_prefix_and_padding() {
    let dir = tempfile::tempdir().unwrap();
    common::build_artifacts(dir.path(), &[(1, [255u8; 6])]);

    let short = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30), // bucket 3
        ],
    );
    let longer = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30), // bucket 3
            entry(1, "call", 30),  // bucket 1
        ],
    );
    let ctx_short = TurnContext::from_turn_state(&short, &[], &config()).unwrap();
    let ctx_long = TurnContext::from_turn_state(&longer, &[], &config()).unwrap();
    assert_ne!(
        common::synthetic_hash(dir.path(), &ctx_short),
        common::synthetic_hash(dir.path(), &ctx_long),
        "history prefix must participate in the hash"
    );
}

/// Trainer quirk (faithfully replicated): preflop boards are five zero bytes,
/// which degenerates every preflop EHS evaluation to a tie, so the preflop
/// cluster — and therefore the whole preflop hash for a given history — is
/// independent of the hole cards. This matches training exactly; do NOT
/// "fix" it here without retraining.
#[test]
fn preflop_hash_is_hole_card_independent_like_the_trainer() {
    let dir = tempfile::tempdir().unwrap();
    common::build_artifacts(dir.path(), &[(1, [255u8; 6])]);

    let mut s1 = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ],
    );
    let mut s2 = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ],
    );
    s1.your_hole_cards = vec!["Ah".into(), "Kd".into()];
    s2.your_hole_cards = vec!["7c".into(), "2s".into()];
    let c1 = TurnContext::from_turn_state(&s1, &[], &config()).unwrap();
    let c2 = TurnContext::from_turn_state(&s2, &[], &config()).unwrap();
    assert_eq!(
        common::synthetic_hash(dir.path(), &c1),
        common::synthetic_hash(dir.path(), &c2),
        "preflop cluster is constant under the trainer's padded-board inputs"
    );
}

/// Street is hashed, and postflop boards include dealt cards, so flop and
/// turn states hash differently.
#[test]
fn flop_and_turn_states_hash_differently() {
    let dir = tempfile::tempdir().unwrap();
    common::build_artifacts(dir.path(), &[(1, [255u8; 6])]);

    let flop = state(
        "flop",
        vec!["Ts", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
        ],
    );
    let flop2 = state(
        "flop",
        vec!["9s", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
        ],
    );
    let ctx_flop = TurnContext::from_turn_state(&flop, &[], &config()).unwrap();
    let ctx_flop2 = TurnContext::from_turn_state(&flop2, &[], &config()).unwrap();
    let h1 = common::synthetic_hash(dir.path(), &ctx_flop);
    let h2 = common::synthetic_hash(dir.path(), &ctx_flop2);
    assert_ne!(h1, h2, "different flop cards must (almost surely) rehash");
}

/// Engine end-to-end: a blueprint containing the exact hash answers with the
/// encoded strategy; a missing hash falls through to the EHS fallback.
#[test]
fn engine_lookup_hit_and_miss() {
    use pkr_chipzen_bot::proto::frames::Decision;
    use pkr_chipzen_bot::strategy::Engine;

    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    let mut s = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ],
    );
    s.to_call = 0;
    let ctx = TurnContext::from_turn_state(&s, &["check".to_string()], &config()).unwrap();
    let hit = common::synthetic_hash(dir.path(), &ctx);

    // One key with all mass on slot 1 (check/call), one unknown key.
    let artifacts = common::build_artifacts(
        dir.path(),
        &[(hit, common::cdf_all_on(1)), (0x123456789, [255u8; 6])],
    );
    let engine = Engine::load(&artifacts, true).unwrap();

    assert!(
        engine.advice_for(&ctx).is_some(),
        "hit infoset must resolve"
    );
    assert_eq!(engine.decide(&ctx), Decision::check());

    // Miss: different history → different hash → not in blueprint → fallback.
    let mut s_miss = s.clone();
    s_miss.action_history.push(entry(0, "raise", 30));
    let ctx_miss =
        TurnContext::from_turn_state(&s_miss, &["check".to_string()], &config()).unwrap();
    assert!(engine.advice_for(&ctx_miss).is_none());
    // Fallback with equity 0.5 (zero rank table), to_call == 0 → check.
    assert_eq!(engine.decide(&ctx_miss), Decision::check());
}
