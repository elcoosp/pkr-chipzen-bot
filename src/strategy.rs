//! Strategy engine: engine-canonical blueprint decisions (+ optional
//! river subgame search) mapped onto chipzen wire actions.
//!
//! Decision pipeline for one `turn_request`:
//!
//! ```text
//! TurnContext ──► replay_hand (engine GameState, seat-mapped, chip-scaled)
//!             ──► infoset_signature_into  (sig v1..v5, env-selected)
//!             ──► get_infoset_hash        (pkr-abstraction, trainer-exact)
//!             ──► strategy over K=7 buckets
//!                   ├─ session mode (--ckpt): RuntimeSession =
//!                   │   range tracker + river-only bounded vsearch
//!                   │   (PKR_VSEARCH=1) with checkpoint avg-strategy fallback
//!                   └─ blueprint mode (default): SolverHandle.get_advice_fast
//!                       → CDF decode; miss → fallback_advice (or --fallback-ehs)
//!             ──► legal_actions_into + first_legal_by_bucket
//!             ──► sample_legal_index (or argmax with --deterministic)
//!             ──► engine action → wire chips (÷ scale)
//!             ──► legality validation (never send an illegal action)
//! ```
//!
//! Every engine-facing step calls the engine's own functions — the same
//! code paths `pkr-cfr` training and `pkr-bot-pipe` use — so decisions stay
//! byte-exact with training semantics by construction.

use rand::RngExt;

use pkr_abstraction::{load_centroids, KMeansAbstraction};
use pkr_contracts::{AbstractionBuilder, Evaluator, SotaAdvice};
use pkr_cfr::table::CompactRegretTable;
use pkr_eval::TableEvaluator;
use pkr_runtime::session::RuntimeSession;
use pkr_runtime::subgame::{SubgameConfig, SubgameHandle};
use pkr_runtime::{lookup::SolverHandle, MmapReader};

use crate::adapter::{replay_hand, Replay, ReplayEvent, TurnContext};
use crate::error::{BotError, Result};
use crate::proto::frames::Decision;

/// Paths of the trained artifacts.
#[derive(Debug, Clone, Default)]
pub struct ArtifactPaths {
    /// `blueprint.bin` (mmap'd PKRSOTA1 file). Required unless a checkpoint
    /// is provided (the session path reads the average strategy from the
    /// checkpoint itself).
    pub blueprint: String,
    /// `hand_ranks.bin` — OPTIONAL. When absent the exact `NlheEvaluator`
    /// is used (the engine's own reference path, no table file needed).
    /// When present, the mmap `TableEvaluator` is used (faster EHS).
    pub rank_table: Option<String>,
    /// `centroids.bin` (EHS² centroid store, bincode) — required.
    pub centroids: String,
    /// `flop_buckets.bin` (optional; loader kept for artifact parity).
    pub flop_buckets: Option<String>,
    /// Per-street centroid overrides (optional).
    pub flop_centroids: Option<String>,
    pub turn_centroids: Option<String>,
    pub river_centroids: Option<String>,
    /// Street abstraction tables consumed by `get_infoset_hash`:
    /// flop `abstraction.bin`, `turn_abstraction.bin`, `river_buckets.bin`.
    /// The hash consults these on every postflop decision — a missing table
    /// silently reroutes lookups through the Monte-Carlo fallback cluster
    /// (a different infoset key space). Load failures are hard errors;
    /// absence is allowed only with `EngineOptions::require_tables = false`
    /// (test mode) and always logs a warning.
    pub preflop_table: Option<String>,
    pub flop_table: Option<String>,
    pub turn_table: Option<String>,
    pub river_table: Option<String>,
    /// `train.ckpt` — optional session-mode checkpoint (RuntimeSession:
    /// opponent range tracking + river-only bounded search under
    /// `PKR_VSEARCH=1`). ~100 bb k7-baseline checkpoint size ≈ 1.9 GB.
    pub ckpt: Option<String>,
}

/// Engine construction options beyond the artifact paths.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Argmax instead of CDF sampling (deterministic, debuggable,
    /// exploitable over many hands).
    pub deterministic: bool,
    /// Use the EHS heuristic on blueprint misses instead of the engine's
    /// `fallback_advice()` mild policy.
    pub fallback_ehs: bool,
    /// Preflop centroid count for the fingerprint check
    /// (`AbstractionFingerprint::from_constants`). Must match training
    /// (k7-baseline: 200).
    pub preflop_k: u32,
    /// Fail hard when the flop/turn/river abstraction tables are missing.
    /// Tests pass `false` (the synthetic 305 MB turn table would dominate).
    pub require_tables: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            deterministic: false,
            fallback_ehs: false,
            preflop_k: 200,
            require_tables: true,
        }
    }
}

/// Loaded engine: mmap'd blueprint (+ optional checkpoint session),
/// abstraction tables and evaluator.
pub struct Engine {
    solver: Option<SolverHandle>,
    abstraction: std::sync::Arc<KMeansAbstraction>,
    abstraction_dyn: std::sync::Arc<dyn AbstractionBuilder>,
    evaluator: std::sync::Arc<dyn Evaluator>,
    /// Session-mode resources (checkpoint table) — `Some` when `--ckpt`.
    session_table: Option<std::sync::Arc<CompactRegretTable>>,
    deterministic: bool,
    fallback_ehs: bool,
    /// Infosets served from the blueprint so far (health metric).
    pub hits: std::sync::atomic::AtomicU64,
    /// Blueprint misses so far.
    pub misses: std::sync::atomic::AtomicU64,
}

impl Engine {
    /// Load all artifacts. The `PKR_SIG_*` env must already be applied
    /// (see `main.rs`) — the fingerprint and every infoset hash depend on
    /// it, and the engine caches the resolved version in a `OnceLock`.
    pub fn load(paths: &ArtifactPaths, opts: &EngineOptions) -> Result<Engine> {
        if paths.centroids.trim().is_empty() {
            return Err(BotError::Artifact("centroids path is required".into()));
        }

        // Evaluator: mmap table when the rank table exists, else the exact
        // in-process NlheEvaluator (the engine's own reference bot runs it).
        let evaluator: std::sync::Arc<dyn Evaluator> = match &paths.rank_table {
            Some(p) => std::sync::Arc::new(
                TableEvaluator::new(p).map_err(|e| BotError::Artifact(format!("rank table {p}: {e}")))?,
            ),
            None => {
                tracing::info!("no --rank-table: using the exact NlheEvaluator (no hand_ranks.bin needed)");
                std::sync::Arc::new(pkr_eval::NlheEvaluator)
            }
        };

        let store = load_centroids(&paths.centroids)
            .map_err(|e| BotError::Artifact(format!("centroids {}: {e}", paths.centroids)))?;
        tracing::info!(centroids = store.centroids.len(), "centroid store loaded");
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

        // Street tables: the postflop hash arms read these on every call.
        // Flop (1) / turn (2) / river (3) are load-bearing for a k7-baseline
        // blueprint; the preflop table (0) is bypassed by the exact
        // 169-class preflop path.
        let mut missing: Vec<&str> = Vec::new();
        for (street, name, path) in [
            (0u8, "preflop", &paths.preflop_table),
            (1, "flop", &paths.flop_table),
            (2, "turn", &paths.turn_table),
            (3, "river", &paths.river_table),
        ] {
            match path {
                Some(p) => {
                    abstraction.init_table(street, p).map_err(|e| {
                        BotError::Artifact(format!("{name} table {p}: {e}"))
                    })?;
                    tracing::info!(street, table = %p, "abstraction table loaded");
                }
                None => {
                    if street >= 1 {
                        missing.push(match street {
                            1 => "--flop-table (abstraction.bin)",
                            2 => "--turn-table (turn_abstraction.bin)",
                            _ => "--river-table (river_buckets.bin)",
                        });
                    }
                }
            }
        }
        if !missing.is_empty() {
            let list = missing.join(", ");
            if opts.require_tables {
                return Err(BotError::Artifact(format!(
                    "missing street abstraction tables: {list}. Without them \
                     postflop hashes silently reroute through the Monte-Carlo \
                     fallback cluster (a different infoset key space) and the \
                     blueprint would miss. Export them with the training run \
                     (run.sh) and pass the flags."
                )));
            }
            tracing::warn!(
                tables = %list,
                "street tables missing; postflop lookups use the MC fallback cluster (test mode only)"
            );
        }

        // Blueprint (optional in session-only mode).
        let solver = if paths.blueprint.trim().is_empty() {
            if paths.ckpt.is_none() {
                return Err(BotError::Artifact(
                    "either --blueprint or --ckpt is required".into(),
                ));
            }
            tracing::info!("no --blueprint: decisions come from the checkpoint (session mode)");
            None
        } else {
            let reader = MmapReader::new(&paths.blueprint).map_err(|e| {
                BotError::Artifact(format!("blueprint {}: {e}", paths.blueprint))
            })?;
            let hdr = reader.file_header();
            tracing::info!(
                infosets = hdr.infoset_count,
                max_actions_k = hdr.max_actions_k,
                version = hdr.version,
                "blueprint loaded"
            );
            if hdr.max_actions_k as usize != pkr_core::abstraction::NUM_ACTION_BUCKETS {
                return Err(BotError::Artifact(format!(
                    "blueprint K={} but this engine build uses K={} ({}). \
                     The blueprint and the binary must come from the same \
                     abstraction generation.",
                    hdr.max_actions_k,
                    pkr_core::abstraction::NUM_ACTION_BUCKETS,
                    "K=7 since 2026-10-10"
                )));
            }
            let solver = SolverHandle::new(reader);
            // C6: refuse mismatched abstraction fingerprints loudly — a
            // mismatch makes EVERY lookup a miss (silent weak play).
            let fp = pkr_core::abstraction::AbstractionFingerprint::from_constants(
                opts.preflop_k,
            );
            if let Err(e) = solver.check_fingerprint(&fp) {
                return Err(BotError::Artifact(format!(
                    "blueprint fingerprint mismatch ({}). The blueprint was \
                     trained under a different abstraction (preflop_k={}, \
                     sig version, stack depth, …). Pass the matching \
                     --preflop-k / --sig or re-export.",
                    e, opts.preflop_k
                )));
            }
            tracing::info!(
                preflop_k = opts.preflop_k,
                "blueprint fingerprint verified"
            );
            Some(solver)
        };

        // Session mode (optional).
        let session_table = match &paths.ckpt {
            Some(p) => {
                let table = CompactRegretTable::with_capacity(60_000_000);
                let fp = pkr_core::abstraction::AbstractionFingerprint::from_constants(
                    opts.preflop_k,
                );
                let count = table
                    .load_checkpoint(p, &fp)
                    .map_err(|e| BotError::Artifact(format!("checkpoint {p}: {e}")))?;
                tracing::info!(infosets = count, checkpoint = %p, "session checkpoint loaded");
                Some(std::sync::Arc::new(table))
            }
            None => None,
        };

        let abstraction = std::sync::Arc::new(abstraction);

        Ok(Engine {
            abstraction_dyn: abstraction.clone(),
            abstraction,
            evaluator,
            solver,
            session_table,
            deterministic: opts.deterministic,
            fallback_ehs: opts.fallback_ehs,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// True when river subgame search will be attempted (session mode +
    /// `PKR_VSEARCH=1`).
    pub fn vsearch_active(&self) -> bool {
        self.session_table.is_some()
            && std::env::var("PKR_VSEARCH").as_deref() == Ok("1")
    }

    /// Full decision for one turn. Always returns a legal action.
    pub fn decide(&self, ctx: &TurnContext) -> Decision {
        // 1. Replay the wire hand into the engine's coordinate system.
        let rep = match replay_hand(&ctx.state, &ctx.config) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "replay failed; safe wire fallback");
                return check_call(ctx);
            }
        };

        // 2. Trainer-exact infoset hash from the replayed state.
        let st = &rep.state;
        let hole = st.hole[rep.our_seat as usize];
        let sig = crate::adapter::infoset_signature(st);
        let hash = self.abstraction.get_infoset_hash(
            &hole,
            &st.board[..st.board_len as usize],
            &sig,
            st.street as u8,
        );

        // 3. Bucket strategy over K=7. A miss in EHS-fallback mode returns
        // the sentinel (real decoded probabilities are always >= 0).
        let probs: [f64; 7] = if let Some(table) = self.session_table.as_ref() {
            self.session_strategy(&rep, table, hash)
                .unwrap_or_else(|| self.blueprint_strategy(hash))
        } else {
            self.blueprint_strategy(hash)
        };
        if probs == EHS_FALLBACK_SENTINEL {
            let d = self.ehs_fallback(ctx);
            return validate(ctx, d);
        }

        // 4. Engine-legal candidates and bucket → action map.
        let mut legal = [pkr_core::state::Action {
            player: 0,
            kind: pkr_core::state::ActionKind::Fold,
        }; 8];
        let n = st.legal_actions_into(&mut legal);
        let b2a = pkr_core::abstraction::first_legal_by_bucket(
            &legal[..n],
            st.stacks[st.actor],
            st.street_bets[st.actor],
            st.street_bets[1 - st.actor],
            st.pot,
        );

        // 5. Pick a bucket (sample or argmax), then its legal action.
        let chosen = if self.deterministic {
            argmax_present_bucket(&probs, &b2a)
        } else {
            pkr_core::abstraction::sample_legal_index(&probs, &b2a, rand::rng().random::<f64>())
        };
        let decision = match chosen.map(|i| legal[i as usize].kind) {
            Some(pkr_core::state::ActionKind::Fold) => Decision::fold(),
            Some(pkr_core::state::ActionKind::Check) => Decision::check(),
            Some(pkr_core::state::ActionKind::Call) => Decision::call(),
            Some(pkr_core::state::ActionKind::Bet(total)) => {
                // Engine chips → wire chips (raise-to total).
                let wire = (total / rep.scale).round() as i64;
                Decision::raise(wire)
            }
            None => {
                tracing::warn!("no legal bucket candidate; safe wire fallback");
                check_call(ctx)
            }
        };

        tracing::debug!(
            street = st.street as u8,
            seat = rep.our_seat,
            engine_pot = st.pot,
            probs = ?probs,
            action = %decision.action,
            amount = decision.amount,
            "engine decision"
        );

        // 6. Final wire-legality gate.
        validate(ctx, decision)
    }

    /// Blueprint-only bucket strategy (CDF decode; engine fallback policy
    /// or EHS heuristic on a miss).
    fn blueprint_strategy(&self, hash: u64) -> [f64; 7] {
        use std::sync::atomic::Ordering::Relaxed;
        let advice: SotaAdvice = match self.solver.as_ref() {
            Some(solver) => match solver.get_advice_fast(hash) {
                Some(a) => {
                    self.hits.fetch_add(1, Relaxed);
                    a
                }
                None => {
                    self.misses.fetch_add(1, Relaxed);
                    tracing::debug!(hash = %hash, "blueprint miss → fallback policy");
                    if self.fallback_ehs {
                        // The EHS heuristic needs the wire context; signal
                        // via the sentinel and let decide() compute it.
                        return EHS_FALLBACK_SENTINEL;
                    }
                    solver.fallback_advice()
                }
            },
            None => {
                // Session-only mode: the checkpoint avg strategy is the
                // fallback inside advise_or_blueprint; reaching here means
                // the session path returned None entirely — use uniform.
                self.misses.fetch_add(1, Relaxed);
                let mut u = [0.0f64; 7];
                for v in u.iter_mut() {
                    *v = 1.0 / 7.0;
                }
                return u;
            }
        };
        decode_cdf(&advice)
    }

    /// Session-mode strategy: RuntimeSession (range tracker + river-only
    /// bounded vsearch under PKR_VSEARCH=1) with the checkpoint's average
    /// strategy as the in-method fallback.
    fn session_strategy(
        &self,
        rep: &Replay,
        table: &std::sync::Arc<CompactRegretTable>,
        hash: u64,
    ) -> Option<[f64; 7]> {
        let handle = SubgameHandle::new(SubgameConfig {
            evaluator: self.evaluator.clone(),
            abstraction: self.abstraction_dyn.clone(),
            table: table.clone(),
            iters: 25,
            hands_per_range: 8,
            // Search is river-only live: a turn bound decision costs
            // ~45–52 s (offline-only per the engine docs).
            enabled_streets: [false, false, false, true],
        });
        let mut sess = RuntimeSession::new(
            handle,
            rep.our_seat,
            self.abstraction_dyn.as_ref(),
            table,
            self.evaluator.as_ref(),
        );
        sess.deal_start(rep.root.clone());
        for ev in &rep.events {
            match ev {
                ReplayEvent::Act(a) => sess.observe_action(a.clone()),
                ReplayEvent::Street(cards) => sess.observe_street(cards),
            }
        }
        let st = &rep.state;
        let hole = st.hole[rep.our_seat as usize];
        let strat = sess.advise_or_blueprint(st, &hole, hash)?;
        let mut out = [0.0f64; 7];
        out.copy_from_slice(&strat[..7]);
        Some(out)
    }

    /// EHS heuristic fallback (opt-in via `--fallback-ehs`): equity-based
    /// pot-odds play for infosets the blueprint never visited.
    fn ehs_fallback(&self, ctx: &TurnContext) -> Decision {
        let equity =
            pkr_abstraction::calculate_ehs(&ctx.hole, &ctx.board, self.evaluator.as_ref()).0 as f64;
        tracing::debug!(equity, "EHS fallback");
        if ctx.to_call <= 0 {
            if ctx.has_action("raise") && ctx.max_raise > 0 && equity >= 0.62 {
                let f = if equity >= 0.85 { 1.0 } else { 0.5 };
                let amt = clamp_raise(f * ctx.pot.max(1) as f64, ctx);
                return Decision::raise(amt);
            }
            return check_call(ctx);
        }
        let required = ctx.to_call as f64 / (ctx.pot + ctx.to_call).max(1) as f64;
        if equity >= 0.70 && ctx.has_action("raise") && ctx.max_raise > 0 {
            let amt = clamp_raise(1.0 * ctx.pot.max(1) as f64, ctx);
            return Decision::raise(amt);
        }
        let fold_threshold = (0.30f64).max(required - 0.03);
        if equity < fold_threshold && ctx.has_action("fold") {
            return Decision::fold();
        }
        check_call(ctx)
    }
}

/// Sentinel returned by `blueprint_strategy` when the EHS fallback is
/// requested on a miss (decide() then consults the wire context).
const EHS_FALLBACK_SENTINEL: [f64; 7] = [-1.0; 7];

/// Decode a quantized CDF into per-bucket probabilities.
pub fn decode_cdf(advice: &SotaAdvice) -> [f64; 7] {
    let mut probs = [0.0f64; 7];
    let mut prev = 0u16;
    let len = (advice.len as usize).min(7);
    for (i, slot) in probs.iter_mut().enumerate().take(len) {
        let c = advice.cdf_probabilities[i] as u16;
        *slot = (c.saturating_sub(prev)) as f64 / 255.0;
        prev = c;
    }
    probs
}

/// Deterministic bucket choice: highest-probability bucket that has a legal
/// action; ties break to the lowest bucket. If every present bucket has
/// zero probability the first present bucket is chosen.
fn argmax_present_bucket(probs: &[f64; 7], b2a: &[Option<u8>; 7]) -> Option<u8> {
    let mut best: Option<(usize, f64)> = None;
    for b in 0..7 {
        if b2a[b].is_none() {
            continue;
        }
        let p = probs[b];
        if best.map_or(true, |(_, bp)| p > bp) {
            best = Some((b, p));
        }
    }
    let (b, _) = best?;
    b2a[b]
}

fn clamp_raise(raw: f64, ctx: &TurnContext) -> i64 {
    let min = if ctx.min_raise > 0 { ctx.min_raise } else { 1 };
    let max = ctx.max_raise.max(min) as f64;
    raw.round().clamp(min as f64, max) as i64
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
        TurnContext {
            hand_number: 1,
            street: 1,
            hole: [25, 37],
            board: vec![8, 18, 26],
            pot,
            your_stack: max_raise.max(0),
            opponent_stacks: vec![950],
            to_call,
            min_raise,
            max_raise,
            valid_actions: valid.iter().map(|s| s.to_string()).collect(),
            config: GameConfig {
                variant: "nlhe".into(),
                starting_stack: 1000,
                small_blind: 5,
                big_blind: 10,
                ante: 0,
                num_players: 2,
            },
            state: crate::proto::frames::TurnState {
                hand_number: 1,
                phase: "flop".into(),
                board: vec!["Ts".into(), "7h".into(), "2d".into()],
                your_hole_cards: vec!["Ah".into(), "Kd".into()],
                pot,
                your_stack: max_raise.max(0),
                opponent_stacks: vec![950],
                to_call,
                min_raise,
                max_raise,
                action_history: vec![],
            },
        }
    }

    #[test]
    fn cdf_decode_splits_cumulative_bytes() {
        // K=7 CDF: deltas 26, 102, 51, 25, 26, 18, 7 (sum 255).
        let mut full = [0u8; 16];
        let cdf = [26, 128, 179, 204, 230, 248, 255];
        full[..7].copy_from_slice(&cdf);
        let a = SotaAdvice {
            cdf_probabilities: full,
            len: 7,
        };
        let p = decode_cdf(&a);
        assert!((p[0] - 26.0 / 255.0).abs() < 1e-12);
        assert!((p[1] - 102.0 / 255.0).abs() < 1e-12);
        assert!((p[6] - 7.0 / 255.0).abs() < 1e-12);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn argmax_prefers_largest_probability_mass() {
        let mut b2a: [Option<u8>; 7] = [None; 7];
        for (i, b) in b2a.iter_mut().enumerate() {
            *b = Some(i as u8);
        }
        // probs: bucket 3 largest.
        let probs = [0.05, 0.10, 0.20, 0.40, 0.10, 0.10, 0.05];
        assert_eq!(argmax_present_bucket(&probs, &b2a), Some(3));
        // Ties break to the lowest bucket.
        let tie = [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(argmax_present_bucket(&tie, &b2a), Some(0));
        // Illegal buckets are skipped: only fold/call present, call heavier.
        let mut fold_call: [Option<u8>; 7] = [None; 7];
        fold_call[0] = Some(0);
        fold_call[1] = Some(1);
        assert_eq!(argmax_present_bucket(&probs, &fold_call), Some(1));
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

    #[test]
    fn check_call_prefers_free_check_and_facing_call() {
        let free = ctx(&["check", "fold", "raise"], 0, 20, 970, 80);
        assert_eq!(check_call(&free), Decision::check());
        let facing = ctx(&["fold", "call", "raise"], 20, 40, 970, 80);
        assert_eq!(check_call(&facing), Decision::call());
    }
}
