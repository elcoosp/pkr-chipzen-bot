//! CLI entrypoint — mirrors the Python reference client's surface:
//!
//! ```text
//! pkr-chipzen-bot --base-url wss://staging.chipzen.ai \
//!                 --bot-id <uuid> \
//!                 --token cz_extbot_... \
//!                 --blueprint artifacts/blueprint.bin \
//!                 --centroids artifacts/centroids.bin \
//!                 --flop-table artifacts/abstraction.bin \
//!                 --turn-table artifacts/turn_abstraction.bin \
//!                 --river-table artifacts/river_buckets.bin \
//!                 [--ckpt artifacts/train.ckpt] \
//!                 [--sig 5] [--preflop-k 200] \
//!                 [--loop] [-v|-vv|-vvv] [--deterministic]
//! ```
//!
//! Environment fallbacks: `CHIPZEN_BASE_URL`, `CHIPZEN_BOT_ID`,
//! `CHIPZEN_EXTBOT_TOKEN` (CLI flags take precedence).
//!
//! Engine runtime env (read once per process — the flags below apply them
//! before anything loads): `PKR_SIG_V5` (via `--sig`), `PKR_VSEARCH`,
//! `PKR_VSEARCH_BUDGET_MS`, `PKR_STACK_BB`. The shipped k7-baseline
//! configuration is `--sig 5` + `PKR_STACK_BB=100` (+ `PKR_VSEARCH=1` in
//! session mode; river-only, proven bound).

// See lib.rs — BotError embeds the (large) tungstenite error enum.
#![allow(clippy::result_large_err)]

use std::path::PathBuf;

use clap::{ArgAction, Parser};

use pkr_chipzen_bot::strategy::ArtifactPaths;
use pkr_chipzen_bot::{run, BotConfig};

#[derive(Parser, Debug)]
#[command(
    name = "pkr-chipzen-bot",
    version,
    about = "Chipzen.ai External-API poker bot powered by the pkr-sota CFR runtime"
)]
struct Cli {
    /// Platform origin, e.g. wss://staging.chipzen.ai
    #[arg(long, env = "CHIPZEN_BASE_URL")]
    base_url: String,

    /// The external-API bot's UUID
    #[arg(long, env = "CHIPZEN_BOT_ID")]
    bot_id: String,

    /// cz_extbot_… token (long-lived; prefer the env var over the flag)
    #[arg(long, env = "CHIPZEN_EXTBOT_TOKEN")]
    token: String,

    /// Keep playing: reconnect the lobby after each match ends
    #[arg(long = "loop")]
    loop_matches: bool,

    /// Verbosity: -v info+debug, -vv trace (RUST_LOG overrides)
    #[arg(short = 'v', action = ArgAction::Count, default_value_t = 0)]
    verbose: u8,

    /// Blueprint file (mmap'd PKRSOTA1 artifact)
    #[arg(long, default_value = "blueprint.bin")]
    blueprint: PathBuf,

    /// 5-card hand rank lookup table (OPTIONAL — when omitted the exact
    /// NlheEvaluator is used and no hand_ranks.bin is needed)
    #[arg(long)]
    rank_table: Option<PathBuf>,

    /// EHS² centroid store (bincode)
    #[arg(long, default_value = "centroids.bin")]
    centroids: PathBuf,

    /// Flop bucket table (optional; artifact parity with the trainer)
    #[arg(long)]
    flop_buckets: Option<PathBuf>,

    /// Per-street centroid overrides (optional)
    #[arg(long)]
    flop_centroids: Option<PathBuf>,
    #[arg(long)]
    turn_centroids: Option<PathBuf>,
    #[arg(long)]
    river_centroids: Option<PathBuf>,

    /// Flop abstraction table (`abstraction.bin` from run.sh) — REQUIRED
    /// for postflop hash parity with the shipped blueprint
    #[arg(long)]
    flop_table: Option<PathBuf>,

    /// Turn abstraction table (`turn_abstraction.bin`) — REQUIRED
    #[arg(long)]
    turn_table: Option<PathBuf>,

    /// River board-bucket table (`river_buckets.bin`) — REQUIRED
    #[arg(long)]
    river_table: Option<PathBuf>,

    /// Preflop abstraction table (optional; the exact 169-class preflop
    /// path bypasses it under the default PKR_PREFLOP_EXACT=1)
    #[arg(long)]
    preflop_table: Option<PathBuf>,

    /// Training checkpoint (`train.ckpt`) — enables SESSION mode:
    /// opponent range tracking + river-only bounded subgame search when
    /// PKR_VSEARCH=1 (the shipped safe search config). Not required for
    /// blueprint-only play.
    #[arg(long)]
    ckpt: Option<PathBuf>,

    /// Infoset signature generation (1–5). The shipped k7-baseline was
    /// trained with PKR_SIG_V5=1 → the default is 5. Must match training
    /// or every lookup misses (the fingerprint check refuses loudly).
    #[arg(long, default_value_t = 5)]
    sig: u8,

    /// Preflop centroid count for the fingerprint check. Must match the
    /// `centroids.bin` the blueprint was trained with (k7-baseline: 200).
    #[arg(long, default_value_t = 200)]
    preflop_k: u32,

    /// EHS Monte-Carlo samples for the opt-in EHS miss fallback. Does NOT
    /// affect infoset hashes (the tables answer cluster lookups).
    #[arg(long)]
    ehs_samples: Option<usize>,

    /// On a blueprint miss use the EHS pot-odds heuristic instead of the
    /// engine's `fallback_advice()` mild policy
    #[arg(long)]
    fallback_ehs: bool,

    /// Pick the argmax action instead of sampling the blueprint CDF
    /// (deterministic, easier to debug; exploitable over many hands)
    #[arg(long)]
    deterministic: bool,
}

fn init_tracing(verbose: u8) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = match verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        };
        EnvFilter::new(format!("pkr_chipzen_bot={level},pkr_sota=warn,pkr_runtime=warn"))
    });
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    // Engine env must be pinned before anything loads (OnceLocks).
    pkr_chipzen_bot::apply_sig_env(cli.sig);

    // The engine's EHS sampler caches the sample count on first use
    // (OnceLock in pkr-abstraction/src/ehs.rs). Apply the flag before the
    // engine is loaded so runtime clusters match training-time clusters.
    if let Some(n) = cli.ehs_samples {
        if std::env::var_os("EHS_SAMPLES").is_none() {
            std::env::set_var("EHS_SAMPLES", n.to_string());
        }
    }

    let artifacts = ArtifactPaths {
        blueprint: cli.blueprint.display().to_string(),
        rank_table: cli.rank_table.map(|p| p.display().to_string()),
        centroids: cli.centroids.display().to_string(),
        flop_buckets: cli.flop_buckets.map(|p| p.display().to_string()),
        flop_centroids: cli.flop_centroids.map(|p| p.display().to_string()),
        turn_centroids: cli.turn_centroids.map(|p| p.display().to_string()),
        river_centroids: cli.river_centroids.map(|p| p.display().to_string()),
        preflop_table: cli.preflop_table.map(|p| p.display().to_string()),
        flop_table: cli.flop_table.map(|p| p.display().to_string()),
        turn_table: cli.turn_table.map(|p| p.display().to_string()),
        river_table: cli.river_table.map(|p| p.display().to_string()),
        ckpt: cli.ckpt.map(|p| p.display().to_string()),
    };

    let cfg = BotConfig {
        base_url: cli.base_url,
        bot_id: cli.bot_id,
        token: cli.token,
        loop_matches: cli.loop_matches,
        artifacts,
        deterministic: cli.deterministic,
        sig: cli.sig,
        preflop_k: cli.preflop_k,
        fallback_ehs: cli.fallback_ehs,
        require_tables: true,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    match rt.block_on(run(cfg)) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
