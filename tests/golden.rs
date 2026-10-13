//! Golden / compatibility tests for the trainer-exact infoset hashing.
//!
//! The bot no longer re-implements the trainer's hash math: it replays the
//! wire hand through the engine's own `GameState` and calls the engine's
//! `infoset_signature_into` + `get_infoset_hash`. These tests pin that
//! wiring against the engine-native construction path — if the adapter and
//! the engine ever disagree, they fail loudly here.

mod common;

use pkr_contracts::AbstractionBuilder;
use pkr_chipzen_bot::adapter::{replay_hand, TurnContext};
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

fn entry_phase(seat: i64, action: &str, amount: i64, phase: &str) -> ActionEntry {
    ActionEntry {
        seat,
        action: action.into(),
        amount,
        phase: phase.into(),
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

/// The adapter's replayed state must hash EXACTLY like the same hand
/// constructed natively through the engine API (GameState::new + apply +
/// advance). This is the core parity guarantee of the replay design.
#[test]
fn replay_hash_matches_native_engine_construction() {
    common::pin_sig_env();
    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    // Wire hand (protocol §4 shape): SB raises to 30, BB calls; flop
    // checks; we are BB facing a free check... we make the decision point
    // the BB flop check (actor = BB).
    let s = state(
        "flop",
        vec!["Ts", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
        ],
    );
    let ctx = TurnContext::from_turn_state(&s, &["check".to_string()], &config()).unwrap();
    let rep = replay_hand(&ctx.state, &ctx.config).unwrap();

    // Native engine construction of the identical hand (engine chips):
    // stack 200, blinds 1/2, SB raise-to 6, BB call, street advance, BB check.
    let mut st = pkr_core::state::GameState::new(pkr_core::state::stack_chips(), 1.0, 2.0);
    st.hole[1] = [25, 37]; // Ah Kd for the BB (our seat)
    st.apply_action_in_place(&pkr_core::state::Action {
        player: 0,
        kind: pkr_core::state::ActionKind::Bet(6.0),
    });
    st.apply_action_in_place(&pkr_core::state::Action {
        player: 1,
        kind: pkr_core::state::ActionKind::Call,
    });
    st.advance_street_in_place(&[8, 18, 26]); // Ts 7h 2d
    st.apply_action_in_place(&pkr_core::state::Action {
        player: 1,
        kind: pkr_core::state::ActionKind::Check,
    });

    // Same money, same street, same actor.
    assert_eq!(rep.state.pot, st.pot);
    assert_eq!(rep.state.street_bets, st.street_bets);
    assert_eq!(rep.state.actor, st.actor);
    assert_eq!(rep.state.street, st.street);
    assert_eq!(rep.state.total_raises, st.total_raises);
    assert_eq!(rep.state.abstract_history[..rep.state.abstract_history_len as usize],
               st.abstract_history[..st.abstract_history_len as usize]);

    // Same infoset signature bytes and hash.
    let sig_adapter = pkr_chipzen_bot::adapter::infoset_signature(&rep.state);
    let sig_native = pkr_chipzen_bot::adapter::infoset_signature(&st);
    assert_eq!(sig_adapter, sig_native, "V5 signature must match the engine's");

    let evaluator: std::sync::Arc<pkr_eval::TableEvaluator> = std::sync::Arc::new(
        pkr_eval::TableEvaluator::new(dir.path().join("hand_ranks.bin")).unwrap(),
    );
    let store = pkr_abstraction::load_centroids(
        dir.path().join("centroids.bin").to_str().unwrap(),
    )
    .unwrap();
    let abs = pkr_abstraction::KMeansAbstraction::from_store(store, evaluator);
    let board: &[u8] = &rep.state.board[..rep.state.board_len as usize];
    let hole = rep.state.hole[rep.our_seat as usize];
    let h1 = abs.get_infoset_hash(&hole, board, &sig_adapter, rep.state.street as u8);
    let h2 = abs.get_infoset_hash(&hole, board, &sig_native, st.street as u8);
    assert_eq!(h1, h2, "adapter and native paths must hash identically");
}

/// Under the current engine the preflop cluster is the exact 169-class
/// preflop hand class — hole cards DO participate in preflop hashes.
#[test]
fn preflop_hash_is_hole_card_dependent() {
    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    let mut s1 = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
        ],
    );
    let mut s2 = s1.clone();
    s1.your_hole_cards = vec!["Ah".into(), "Kd".into()];
    s2.your_hole_cards = vec!["7c".into(), "2s".into()];
    let c1 = TurnContext::from_turn_state(&s1, &[], &config()).unwrap();
    let c2 = TurnContext::from_turn_state(&s2, &[], &config()).unwrap();
    assert_ne!(
        common::synthetic_hash(dir.path(), &c1),
        common::synthetic_hash(dir.path(), &c2),
        "preflop hashes must depend on the hole cards (169-class cluster)"
    );
}

/// Street is hashed, and postflop boards include dealt cards, so flop
/// boards participate in the hash.
#[test]
fn flop_boards_hash_differently() {
    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

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
    assert_ne!(
        common::synthetic_hash(dir.path(), &ctx_flop),
        common::synthetic_hash(dir.path(), &ctx_flop2),
        "different flop cards must (almost surely) rehash"
    );
}

/// Preflop history participation: raise vs raise-then-raise produce
/// different signatures (V5 packs the preflop abstract history base-7).
#[test]
fn preflop_history_participates_in_hash() {
    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    // Both histories end at a legal preflop decision point.
    let short = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30), // BB to act, facing 25
        ],
    );
    let longer = state(
        "preflop",
        vec![],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "raise", 60), // SB to act, facing 30
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

/// Hashes are deterministic across engine reloads (same synthetic
/// artifacts, same wire state → same key).
#[test]
fn hash_is_deterministic_across_engine_reloads() {
    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    let s = state(
        "flop",
        vec!["Ts", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
        ],
    );
    let ctx = TurnContext::from_turn_state(&s, &["check".to_string()], &config()).unwrap();
    let h1 = common::synthetic_hash(dir.path(), &ctx);
    let h2 = common::synthetic_hash(dir.path(), &ctx);
    assert_ne!(h1, 0);
    assert_eq!(h1, h2, "synthetic artifacts must hash deterministically");
}

/// Engine end-to-end: a blueprint containing the exact hash answers with
/// the encoded strategy; a missing hash falls through to the engine's
/// `fallback_advice` mild policy (deterministic argmax → call here).
#[test]
fn engine_lookup_hit_and_miss() {
    use pkr_chipzen_bot::proto::frames::Decision;
    use pkr_chipzen_bot::strategy::{Engine, EngineOptions};

    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    let s = state(
        "flop",
        vec!["Ts", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
            entry_phase(1, "check", 0, "flop"),
        ],
    );
    let ctx = TurnContext::from_turn_state(&s, &["check".to_string()], &config()).unwrap();
    let hit = common::synthetic_hash(dir.path(), &ctx);

    // One key with all mass on slot 1 (check/call), one unknown key.
    let artifacts = common::build_artifacts(
        dir.path(),
        &[(hit, common::cdf_all_on(1)), (0x123456789, [255u8; 7])],
    );
    let engine = Engine::load(
        &artifacts,
        &EngineOptions {
            deterministic: true,
            preflop_k: common::SYNTHETIC_PREFLOP_K,
            require_tables: false,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(engine.decide(&ctx), Decision::check());

    // Miss: a different (legal) hand → different hash → not in blueprint →
    // the engine's fallback policy. Preflop BB facing a raise-to-30: the
    // fallback CDF's argmax legal bucket is check/call → call.
    let s_miss = TurnState {
        hand_number: 2,
        phase: "preflop".into(),
        board: vec![],
        your_hole_cards: vec!["Ah".into(), "Kd".into()],
        pot: 40,
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
    let ctx_miss =
        TurnContext::from_turn_state(&s_miss, &["fold", "call"].map(String::from), &config())
            .unwrap();
    assert_eq!(engine.decide(&ctx_miss), Decision::call());
}

/// K=7 slot semantics through the full decide() path: jam slot 6 shoves
/// max_raise; overbet slot 5 raises 3× pot; small-bet slot 2 raises the
/// 0.5× candidate; slot 0 folds.
#[test]
fn k7_slots_map_to_wire_actions() {
    use pkr_chipzen_bot::proto::frames::Decision;
    use pkr_chipzen_bot::strategy::{Engine, EngineOptions};

    let dir = tempfile::tempdir().unwrap();
    common::build_base_artifacts(dir.path());

    // Flop free-check node for the BB: pot 12 engine (60 wire), street
    // bets [0, 0], stacks [194, 194]. Legal ladder (engine chips):
    // 6 (0.5×), 12 (1×), 24 (2×), 36 (3×), jam 194.
    // Wire totals: 30, 60, 120, 180, shove 970.
    let s = state(
        "flop",
        vec!["Ts", "7h", "2d"],
        vec![
            entry(0, "post_small_blind", 5),
            entry(1, "post_big_blind", 10),
            entry(0, "raise", 30),
            entry(1, "call", 30),
        ],
    );
    let ctx =
        TurnContext::from_turn_state(&s, &["check", "raise"].map(String::from), &config())
            .unwrap();
    let hit = common::synthetic_hash(dir.path(), &ctx);

    for (slot, expected) in [
        (2usize, 30i64),
        (3, 60),
        (4, 120),
        (5, 180),
        (6, 970), // jam → max_raise (stack 970)
    ] {
        let artifacts = common::build_artifacts(dir.path(), &[(hit, common::cdf_all_on(slot))]);
        let engine = Engine::load(
            &artifacts,
            &EngineOptions {
                deterministic: true,
                preflop_k: common::SYNTHETIC_PREFLOP_K,
                require_tables: false,
                ..Default::default()
            },
        )
        .unwrap();
        let d = engine.decide(&ctx);
        assert_eq!(
            d,
            Decision::raise(expected),
            "slot {slot} must size to {expected} wire chips"
        );
    }
}
