//! CLI entrypoint — mirrors the Python reference client's surface:
//!
//! ```text
//! pkr-chipzen-bot --base-url wss://staging.chipzen.ai \
//!                 --bot-id <uuid> \
//!                 --token cz_extbot_... \
//!                 [--loop] [-v|-vv|-vvv]
//! ```
//!
//! Environment fallbacks: `CHIPZEN_BASE_URL`, `CHIPZEN_BOT_ID`,
//! `CHIPZEN_EXTBOT_TOKEN` (CLI flags take precedence).
//!
//! Artifact flags mirror `pkr-trainer` (defaults are the trainer's default
//! filenames): `--blueprint blueprint.bin --rank-table hand_ranks.bin
//! --centroids centroids.bin` plus the optional per-street tables.

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

    /// 5-card hand rank lookup table (u32 LE, combinadic index)
    #[arg(long, default_value = "hand_ranks.bin")]
    rank_table: PathBuf,

    /// EHS² centroid store (bincode)
    #[arg(long, default_value = "centroids.bin")]
    centroids: PathBuf,

    /// Flop bucket table (optional)
    #[arg(long)]
    flop_buckets: Option<PathBuf>,

    /// Per-street centroid overrides (optional)
    #[arg(long)]
    flop_centroids: Option<PathBuf>,
    #[arg(long)]
    turn_centroids: Option<PathBuf>,
    #[arg(long)]
    river_centroids: Option<PathBuf>,

    /// Street abstraction tables (optional; loaded for trainer parity)
    #[arg(long)]
    preflop_table: Option<PathBuf>,
    #[arg(long)]
    flop_table: Option<PathBuf>,
    #[arg(long)]
    turn_table: Option<PathBuf>,
    #[arg(long)]
    river_table: Option<PathBuf>,

    /// EHS Monte-Carlo samples per cluster computation. Use the SAME value
    /// your trainer used (run.sh exported EHS_SAMPLES=100; the engine's
    /// default is 1000). Overrides the EHS_SAMPLES env var.
    #[arg(long)]
    ehs_samples: Option<usize>,

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
        EnvFilter::new(format!("pkr_chipzen_bot={level},pkr_sota=warn"))
    });
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

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
        rank_table: cli.rank_table.display().to_string(),
        centroids: cli.centroids.display().to_string(),
        flop_buckets: cli.flop_buckets.map(|p| p.display().to_string()),
        flop_centroids: cli.flop_centroids.map(|p| p.display().to_string()),
        turn_centroids: cli.turn_centroids.map(|p| p.display().to_string()),
        river_centroids: cli.river_centroids.map(|p| p.display().to_string()),
        preflop_table: cli.preflop_table.map(|p| p.display().to_string()),
        flop_table: cli.flop_table.map(|p| p.display().to_string()),
        turn_table: cli.turn_table.map(|p| p.display().to_string()),
        river_table: cli.river_table.map(|p| p.display().to_string()),
    };

    let cfg = BotConfig {
        base_url: cli.base_url,
        bot_id: cli.bot_id,
        token: cli.token,
        loop_matches: cli.loop_matches,
        artifacts,
        deterministic: cli.deterministic,
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
