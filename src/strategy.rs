//! Strategy engine: blueprint lookup + action mapping + EHS fallback.
//!
//! Decision pipeline for one `turn_request`:
//!
//! ```text
//! TurnContext ──► infoset hash (pkr-abstraction, trainer-exact inputs)
//!            ──► SolverHandle.get_advice_fast(hash) ──► SotaAdvice (CDF over K=6)
//!            ──► sample (or argmax with --deterministic)
//!            ──► abstract slot → concrete chipzen action
//!                 (ladder-inverted raise sizing, clamped to [min_raise, max_raise])
//!            ──► legality validation (never send an illegal action)
//! ```
//!
//! On a blueprint miss the EHS heuristic policy decides instead (see
//! [`Engine::fallback_decision`]).
//!
//! The slot→action mapping and the size ladder are *pure functions* so they
//! are unit-testable without any loaded artifacts (mirroring the reference
//! `strategy.py` design, where decide() is side-effect-free).

use rand::RngExt;

use pkr_abstraction::{calculate_ehs, load_centroids, KMeansAbstraction};
use pkr_contracts::{AbstractionBuilder, SotaAdvice};
use pkr_eval::TableEvaluator;
use pkr_runtime::{lookup::SolverHandle, MmapReader};

use crate::adapter::{bucket_of_raise, TurnContext};
use crate::error::{BotError, Result};
use crate::proto::frames::Decision;

/// The trainer's bet-size ladder (`pkr-core/src/state.rs::legal_actions`):
/// raise totals are generated as `to_call + f * pot` (facing a bet) or
/// `f * pot` (opening), for these fractions in this order.
pub const LADDER: [f64; 5] = [0.5, 0.75, 1.0, 1.5, 2.0];

/// Paths of the trained artifacts.
#[derive(Debug, Clone, Default)]
pub struct ArtifactPaths {
    pub blueprint: String,
    pub rank_table: String,
    pub centroids: String,
    pub flop_buckets: Option<String>,
    pub flop_centroids: Option<String>,
    pub turn_centroids: Option<String>,
    pub river_centroids: Option<String>,
    /// Street abstraction tables (preflop/flop/turn/river). Loaded for
    /// completeness with the shipped trainer; note the trainer's hash path
    /// always computes clusters live (the `board.len() == 5` arm), so these
    /// are not consulted by the current engine build. Loading them keeps us
    /// drop-in compatible if a future trainer starts using the table branches.
    pub preflop_table: Option<String>,
    pub flop_table: Option<String>,
    pub turn_table: Option<String>,
    pub river_table: Option<String>,
}

/// Loaded engine: mmap'd blueprint + abstraction + evaluator.
pub struct Engine {
    solver: SolverHandle,
    abstraction: KMeansAbstraction,
    evaluator: std::sync::Arc<TableEvaluator>,
    deterministic: bool,
}

impl Engine {
    /// Load all artifacts. Mirrors `binaries/pkr-trainer/src/main.rs` setup.
    pub fn load(paths: &ArtifactPaths, deterministic: bool) -> Result<Engine> {
        let evaluator = std::sync::Arc::new(
            TableEvaluator::new(&paths.rank_table)
                .map_err(|e| BotError::Artifact(format!("rank table {}: {e}", paths.rank_table)))?,
        );

        let store = load_centroids(&paths.centroids)
            .map_err(|e| BotError::Artifact(format!("centroids {}: {e}", paths.centroids)))?;
        let mut abstraction = KMeansAbstraction::from_store(store, evaluator.clone());

        for (street, path) in [
            (1u8, &paths.flop_centroids),
            (2, &paths.turn_centroids),
            (3, &paths.river_centroids),
        ] {
            if let Some(p) = path {
                abstraction
                    .load_street_centroids(street, p)
                    .map_err(|e| BotError::Artifact(format!("street centroids {p}: {e}")))?;
            }
        }
        if let Some(p) = &paths.flop_buckets {
            abstraction
                .load_flop_buckets(p)
                .map_err(|e| BotError::Artifact(format!("flop buckets {p}: {e}")))?;
        }
        for (street, path) in [
            (0u8, &paths.preflop_table),
            (1, &paths.flop_table),
            (2, &paths.turn_table),
            (3, &paths.river_table),
        ] {
            if let Some(p) = path {
                abstraction
                    .init_table(street, p)
                    .map_err(|e| BotError::Artifact(format!("street table {p}: {e}")))?;
            }
        }

        let reader = MmapReader::new(&paths.blueprint)
            .map_err(|e| BotError::Artifact(format!("blueprint {}: {e}", paths.blueprint)))?;
        tracing::info!(
            infosets = reader.file_header().infoset_count,
            max_actions_k = reader.file_header().max_actions_k,
            version = reader.file_header().version,
            "blueprint loaded",
        );
        let solver = SolverHandle::new(reader);

        Ok(Engine { solver, abstraction, evaluator, deterministic })
    }

    /// The trainer-exact infoset hash for this decision point.
    pub fn infoset_hash(&self, ctx: &TurnContext) -> u64 {
        let board = ctx.padded_board();
        let history = ctx.padded_history();
        self.abstraction
            .get_infoset_hash(&ctx.hole, &board, &history, ctx.street)
    }

    /// Blueprint advice for this infoset, if present.
    pub fn advice_for(&self, ctx: &TurnContext) -> Option<SotaAdvice> {
        self.solver.get_advice_fast(self.infoset_hash(ctx))
    }

    /// Full decision for one turn. Always returns a legal action.
    pub fn decide(&self, ctx: &TurnContext) -> Decision {
        let decision = match self.advice_for(ctx) {
            Some(advice) => {
                let slot = pick_slot(&advice, self.deterministic);
                tracing::debug!(
                    slot,
                    cdf = ?&advice.cdf_probabilities[..advice.len as usize],
                    "blueprint advice"
                );
                map_slot(ctx, slot, self.deterministic)
            }
            None => {
                tracing::debug!("infoset miss → EHS fallback policy");
                self.fallback_decision(ctx)
            }
        };
        validate(ctx, decision)
    }

    /// EHS-based heuristic policy for blueprint misses.
    ///
    /// * free check: bet 0.5–1.0× pot when strong (EHS ≥ 0.62 / 0.85), else check
    /// * facing a bet: fold when equity is far below pot odds, call when
    ///   close, raise when very strong (EHS ≥ 0.70)
    fn fallback_decision(&self, ctx: &TurnContext) -> Decision {
        let equity = calculate_ehs(&ctx.hole, &ctx.board, self.evaluator.as_ref()).0;
        tracing::debug!(equity, "fallback EHS computed");
        if ctx.to_call <= 0 {
            if ctx.has_action("raise") && ctx.max_raise > 0 && equity >= 0.62 {
                let f = if equity >= 0.85 { 1.0 } else { 0.5 };
                let amt = clamp_raise(f * ctx.pot.max(1) as f64, ctx);
                return Decision::raise(amt);
            }
            return check_call(ctx);
        }
        let required = ctx.to_call as f64 / (ctx.pot + ctx.to_call).max(1) as f64;
        if equity as f64 >= 0.70 && ctx.has_action("raise") && ctx.max_raise > 0 {
            let amt = clamp_raise(1.0 * ctx.pot.max(1) as f64, ctx);
            return Decision::raise(amt);
        }
        let fold_threshold = (0.30f64).max(required - 0.03);
        if (equity as f64) < fold_threshold && ctx.has_action("fold") {
            return Decision::fold();
        }
        check_call(ctx)
    }
}

// ---------------------------------------------------------------------------
// Pure mapping functions (no engine state required)
// ---------------------------------------------------------------------------

/// Sample (or argmax) an abstract slot from the advice CDF.
///
/// The CDF is cumulative over `u8` probability space (written by
/// `pkr-export/src/writer.rs`); sampling draws `r ∈ [0, 256)` and takes the
/// first bucket whose cumulative strictly exceeds `r`.
pub fn pick_slot(advice: &SotaAdvice, deterministic: bool) -> usize {
    let len = advice.len as usize;
    let cdf = &advice.cdf_probabilities;
    if deterministic {
        // Argmax over per-action probabilities (ties → lowest index).
        let (mut best, mut best_p, mut prev) = (0usize, 0u8, 0u8);
        for (a, &c) in cdf.iter().take(len).enumerate() {
            let p = c.wrapping_sub(prev);
            if p > best_p {
                best_p = p;
                best = a;
            }
            prev = c;
        }
        best
    } else {
        let mut rng = rand::rng();
        // CDF bytes are cumulative probabilities quantized to 255 levels
        // (`(cumulative * 255).round()` in the writer), with the final entry
        // always 255. Drawing r from 0..=254 (255 equiprobable levels) makes
        // P(slot) = (cdf[a] - cdf[a-1]) / 255 exactly, and r=255 can never
        // leak into the trailing slot.
        let r: u32 = rng.random_range(0..255u32);
        cdf.iter()
            .take(len)
            .position(|&c| c as u32 > r)
            .unwrap_or(len - 1)
    }
}

/// Map an abstract slot (0..6) to a concrete chipzen action.
///
/// Slots that are unavailable at this node (the blueprint can carry mass on
/// actions the trainer's legal-action set lacked at that node) degrade
/// gracefully: bet slots → check/call, fold → check/call, all-in → raise(max)
/// or call.
pub fn map_slot(ctx: &TurnContext, slot: usize, deterministic: bool) -> Decision {
    match slot {
        0 => {
            if ctx.has_action("fold") {
                Decision::fold()
            } else {
                check_call(ctx)
            }
        }
        1 => check_call(ctx),
        2..=4 => {
            if ctx.has_action("raise") && ctx.max_raise > 0 {
                ladder_sized_raise(ctx, slot, deterministic)
                    .unwrap_or_else(|| check_call(ctx))
            } else {
                check_call(ctx)
            }
        }
        5 => {
            if ctx.has_action("raise") && ctx.max_raise > 0 {
                Decision::raise(ctx.max_raise)
            } else {
                check_call(ctx)
            }
        }
        other => {
            tracing::warn!(slot = other, "advice slot out of range");
            check_call(ctx)
        }
    }
}

/// `check` when free, else `call`, else `check`, else `fold`.
pub fn check_call(ctx: &TurnContext) -> Decision {
    if ctx.to_call <= 0 {
        if ctx.has_action("check") {
            return Decision::check();
        }
        if ctx.has_action("call") {
            return Decision::call();
        }
    } else {
        if ctx.has_action("call") {
            return Decision::call();
        }
        if ctx.has_action("check") {
            return Decision::check();
        }
    }
    if ctx.has_action("fold") {
        Decision::fold()
    } else {
        // Degenerate: the server always offers check or fold.
        Decision::check()
    }
}

/// Generate concrete raise totals from the trainer's own size ladder and pick
/// one whose *engine bucket* equals the sampled slot.
///
/// This keeps runtime actions on-tree: the concrete sizes are exactly the
/// sizes the trainer's `legal_actions()` generated, so the chosen
/// (infoset, bucket) pair was actually explored during training. During
/// training the concrete size for a bucket was sampled uniformly among the
/// ladder candidates in that bucket — we mirror that: sampling mode draws a
/// uniform candidate among the best bucket-match set; `deterministic` mode
/// takes the first (smallest ladder size) for reproducibility.
pub fn ladder_sized_raise(
    ctx: &TurnContext,
    slot: usize,
    deterministic: bool,
) -> Option<Decision> {
    debug_assert!((2..=4).contains(&slot));
    let pot = ctx.pot.max(1);
    let my_sb = my_street_bet(ctx);
    let min_raise = if ctx.min_raise > 0 { ctx.min_raise } else { 1 };
    let max_raise = ctx.max_raise;

    let mut candidates: Vec<i64> = Vec::with_capacity(LADDER.len());
    for &f in LADDER.iter() {
        let raw = if ctx.to_call > 0 {
            ctx.to_call as f64 + f * pot as f64
        } else {
            f * pot as f64
        };
        // Round to chips and clamp into the legal raise window.
        let mut amt = raw.round() as i64;
        if amt < min_raise {
            amt = min_raise;
        }
        if amt > max_raise {
            amt = max_raise;
        }
        if !candidates.contains(&amt) {
            candidates.push(amt);
        }
    }

    // Candidates that clamped into an all-in shove are excluded while a
    // non-shove candidate exists, so slots 2–4 never escalate into an
    // accidental jam.
    let bucket_of = |amt: i64| bucket_of_raise(amt, ctx.your_stack, my_sb, ctx.pot);
    let has_non_shove = candidates.iter().any(|&c| bucket_of(c) != 5);
    let scored: Vec<(i64, u8)> = candidates
        .iter()
        .filter(|&&c| bucket_of(c) != 5 || slot == 5 || !has_non_shove)
        .map(|&c| (c, bucket_of(c)))
        .collect();
    let best_dist = scored.iter().map(|(_, b)| (*b as i64 - slot as i64).abs()).min()?;
    let best_set: Vec<i64> = scored
        .iter()
        .filter(|(_, b)| (*b as i64 - slot as i64).abs() == best_dist)
        .map(|(c, _)| *c)
        .collect();
    debug_assert!(!best_set.is_empty());

    let amt = if deterministic || best_set.len() == 1 {
        best_set[0]
    } else {
        use rand::seq::IndexedRandom;
        *best_set.choose(&mut rand::rng())?
    };
    tracing::debug!(amount = amt, slot, "ladder-sized raise");
    Some(Decision::raise(amt))
}

/// Our current street bet. `max_raise = bet_this_round + stack`, so
/// `bet_this_round = max_raise - your_stack` (valid whenever raising is legal).
fn my_street_bet(ctx: &TurnContext) -> i64 {
    if ctx.max_raise > 0 {
        (ctx.max_raise - ctx.your_stack).max(0)
    } else {
        0
    }
}

fn clamp_raise(raw: f64, ctx: &TurnContext) -> i64 {
    let min = if ctx.min_raise > 0 { ctx.min_raise } else { 1 };
    let max = ctx.max_raise.max(min) as f64;
    raw.round().clamp(min as f64, max) as i64
}

/// Final legality gate — the bot must never emit an illegal action.
pub fn validate(ctx: &TurnContext, decision: Decision) -> Decision {
    match decision.action.as_str() {
        "raise" => {
            if !ctx.has_action("raise") || ctx.max_raise <= 0 {
                return check_call(ctx);
            }
            let min = if ctx.min_raise > 0 { ctx.min_raise } else { 1 };
            let amount = decision
                .amount
                .unwrap_or(ctx.max_raise)
                .clamp(min, ctx.max_raise);
            Decision::raise(amount)
        }
        "call" => {
            if ctx.has_action("call") {
                Decision::call()
            } else {
                check_call(ctx)
            }
        }
        "check" => {
            if ctx.has_action("check") {
                Decision::check()
            } else {
                check_call(ctx)
            }
        }
        "fold" => {
            if ctx.has_action("fold") {
                Decision::fold()
            } else {
                check_call(ctx)
            }
        }
        other => {
            tracing::warn!(action = other, "strategy produced unknown action");
            check_call(ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::frames::GameConfig;

    fn ctx(valid: &[&str], to_call: i64, min_raise: i64, max_raise: i64, pot: i64) -> TurnContext {
        ctx_stack(valid, to_call, min_raise, max_raise, pot, max_raise.max(0))
    }

    fn ctx_stack(
        valid: &[&str],
        to_call: i64,
        min_raise: i64,
        max_raise: i64,
        pot: i64,
        your_stack: i64,
    ) -> TurnContext {
        TurnContext {
            hand_number: 1,
            street: 1,
            hole: [25, 37],
            board: vec![8, 18, 26],
            pot,
            your_stack,
            opponent_stacks: vec![950],
            to_call,
            min_raise,
            max_raise,
            valid_actions: valid.iter().map(|s| s.to_string()).collect(),
            abstract_history: vec![1],
            config: GameConfig {
                variant: "nlhe".into(),
                starting_stack: 1000,
                small_blind: 5,
                big_blind: 10,
                ante: 0,
                num_players: 2,
            },
        }
    }

    fn advice(cdf: [u8; 6]) -> SotaAdvice {
        let mut full = [0u8; 16];
        full[..6].copy_from_slice(&cdf);
        SotaAdvice { cdf_probabilities: full, len: 6 }
    }

    #[test]
    fn cdf_sampling_respects_cumulative_mass() {
        // All mass on slot 1.
        let a = advice([0, 255, 255, 255, 255, 255]);
        for _ in 0..200 {
            assert_eq!(pick_slot(&a, false), 1);
        }
        // Half slot 0 / half slot 1.
        let a = advice([128, 255, 255, 255, 255, 255]);
        let mut zeros = 0;
        for _ in 0..2000 {
            if pick_slot(&a, false) == 0 {
                zeros += 1;
            }
        }
        assert!(zeros > 800 && zeros < 1200, "zeros = {zeros}");
    }

    #[test]
    fn argmax_prefers_largest_probability_mass() {
        // per-action probs (deltas): 40, 50, 50, 100, 10, 5 → slot 3
        let a = advice([40, 90, 140, 240, 250, 255]);
        assert_eq!(pick_slot(&a, true), 3);
    }

    #[test]
    fn slot_one_maps_to_check_when_free_and_call_when_facing_bet() {
        let free = ctx(&["check", "fold", "raise"], 0, 20, 970, 80);
        assert_eq!(check_call(&free), Decision::check());
        let facing = ctx(&["fold", "call", "raise"], 20, 40, 970, 80);
        assert_eq!(check_call(&facing), Decision::call());
    }

    #[test]
    fn ladder_sizing_lands_in_sampled_bucket() {
        // Opening bet, pot 80, stack 970 (max_raise = 970, my_sb = 0).
        // Ladder candidates: 40, 60, 80, 120, 160 → buckets 2, 2, 3, 3, 3
        // (post-action fraction amt/(pot+amt) is always < 1, so bucket 4 is
        // unreachable on a fresh opening bet — slot 4 degrades to the
        // nearest-bucket set). Deterministic mode takes the first of the
        // best set, mirroring the trainer's uniform-within-bucket sampling.
        let c = ctx(&["check", "raise"], 0, 20, 970, 80);
        let d2 = ladder_sized_raise(&c, 2, true).expect("slot 2");
        assert_eq!(d2.amount.unwrap(), 40);
        assert_eq!(bucket_of_raise(40, c.your_stack, 0, c.pot), 2);
        let d3 = ladder_sized_raise(&c, 3, true).expect("slot 3");
        assert_eq!(d3.amount.unwrap(), 80);
        assert_eq!(bucket_of_raise(80, c.your_stack, 0, c.pot), 3);
        // Slot 4: nearest available bucket (3) — first of the set.
        let d4 = ladder_sized_raise(&c, 4, true).expect("slot 4");
        assert_eq!(d4.amount.unwrap(), 80);
        assert_eq!(bucket_of_raise(80, c.your_stack, 0, c.pot), 3);
        // Sampling mode must stay inside the best set for the slot.
        for _ in 0..50 {
            let d = ladder_sized_raise(&c, 3, false).unwrap();
            assert_eq!(bucket_of_raise(d.amount.unwrap(), c.your_stack, 0, c.pot), 3);
        }
    }

    #[test]
    fn ladder_respects_min_raise_clamp() {
        // Preflop pot 30, min_raise 20: candidates 15→20, 22.5→23 clamped;
        // buckets 2, 2, 3, 3, 3 → slot 2 takes the first (20).
        let c = ctx(&["check", "raise"], 0, 20, 1000, 30);
        assert_eq!(ladder_sized_raise(&c, 2, true), Some(Decision::raise(20)));
    }

    #[test]
    fn ladder_avoids_accidental_shove_for_bet_slots() {
        // Stack 50 with 70 already in (max_raise = 120) facing 50: every
        // ladder candidate clamps to the 120 jam — slot 4 falls back to it.
        let c = ctx_stack(&["call", "raise"], 50, 100, 120, 200, 50);
        assert_eq!(ladder_sized_raise(&c, 4, true), Some(Decision::raise(120)));
        // But when a non-shove candidate exists, slots 2–4 prefer it.
        let c = ctx(&["call", "raise"], 20, 40, 970, 80);
        for slot in [2usize, 3, 4] {
            let d = ladder_sized_raise(&c, slot, true).unwrap();
            assert!(d.amount.unwrap() < 970, "slot {slot} must not jam");
        }
    }

    #[test]
    fn slot_five_shoves_max_raise() {
        let c = ctx(&["fold", "call", "raise"], 20, 40, 970, 80);
        assert_eq!(map_slot(&c, 5, true), Decision::raise(970));
        // No raise available → degrade to call.
        let c = ctx(&["fold", "call"], 20, 0, 0, 80);
        assert_eq!(map_slot(&c, 5, true), Decision::call());
    }

    #[test]
    fn bet_slots_degrade_to_check_call_without_raise() {
        let facing = ctx(&["fold", "call"], 20, 0, 0, 80);
        assert_eq!(map_slot(&facing, 3, true), Decision::call());
        let free = ctx(&["check", "fold"], 0, 0, 0, 80);
        assert_eq!(map_slot(&free, 3, true), Decision::check());
    }

    #[test]
    fn validate_never_emits_illegal_actions() {
        // raise not offered → falls back to check/call
        let c = ctx(&["fold", "call"], 20, 0, 0, 80);
        assert_eq!(validate(&c, Decision::raise(500)), Decision::call());
        // raise amount clamped into window
        let c = ctx(&["fold", "call", "raise"], 20, 40, 100, 80);
        assert_eq!(validate(&c, Decision::raise(999)), Decision::raise(100));
        assert_eq!(validate(&c, Decision::raise(3)), Decision::raise(40));
        // check not offered when to_call > 0 → call
        let c = ctx(&["fold", "call"], 20, 0, 0, 80);
        assert_eq!(validate(&c, Decision::check()), Decision::call());
        // fold not offered when free → check
        let c = ctx(&["check", "raise"], 0, 20, 100, 80);
        assert_eq!(validate(&c, Decision::fold()), Decision::check());
    }
}
