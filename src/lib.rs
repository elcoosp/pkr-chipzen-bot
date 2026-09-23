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
