//! `pkr-chipzen-bot` — a chipzen.ai External-API poker bot powered by the
//! pkr-sota CFR runtime.
//!
//! Library entry points (also used by the integration tests):
//! * [`run`] — connect the lobby, play matches until told to stop
//! * [`BotConfig`] — runtime configuration
//! * [`strategy::Engine`] — loaded blueprint + abstraction
//!
//! Wire protocol: `docs/EXTERNAL-API-BOT-PROTOCOL.md` (chipzen-sdk) +
//! Layer-1/Layer-2 specs. Blueprint compatibility: `src/adapter.rs` module
//! docs.

// `BotError` embeds `tungstenite::Error`, which is a large but convenient
// enum; boxing it everywhere would hurt readability for zero practical gain
// on this bot's message rates.
#![allow(clippy::result_large_err)]

pub mod adapter;
pub mod config;
pub mod error;
pub mod lobby;
pub mod match_client;
pub mod proto;
pub mod runner;
pub mod strategy;

pub use config::BotConfig;
pub use error::{BotError, Result};
pub use runner::run;
pub use strategy::Engine;

/// Apply the infoset-signature generation to the engine's env dispatch —
/// MUST run before any engine type is constructed (`OnceLock` caches).
/// Mirrors `pkr-runtime/src/bin/pkr-bot-pipe.rs --sig`. Called by `run`
/// with `BotConfig::sig`; `main.rs` additionally applies it before the
/// tokio runtime starts.
pub fn apply_sig_env(sig: u8) {
    match sig {
        5 => {
            std::env::set_var("PKR_SIG_V5", "1");
            std::env::set_var("PKR_SIG_V4", "0");
            std::env::set_var("PKR_SIG_V3", "1");
        }
        4 => {
            std::env::set_var("PKR_SIG_V4", "1");
            std::env::set_var("PKR_SIG_V5", "0");
            std::env::set_var("PKR_SIG_V3", "1");
        }
        3 | 1 => {
            std::env::set_var("PKR_SIG_V3", "1");
            std::env::set_var("PKR_SIG_V4", "0");
            std::env::set_var("PKR_SIG_V5", "0");
        }
        _ => {
            std::env::set_var("PKR_SIG_V3", "0");
            std::env::set_var("PKR_SIG_V4", "0");
            std::env::set_var("PKR_SIG_V5", "0");
        }
    }
}
