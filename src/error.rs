//! Unified error type for the chipzen bot.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BotError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("websocket error: {0}")]
    Ws(#[from] tungstenite::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("artifact error: {0}")]
    Artifact(String),

    #[error("lobby closed before a 'matched' notification arrived")]
    LobbyClosed,

    #[error("lobby evicted us (another connection replaced this bot)")]
    LobbyEvicted,

    #[error("lobby auth rejected (close code {code})")]
    LobbyAuth { code: u16 },

    #[error("match connection closed without a clean match_end (close code {code:?})")]
    MatchClosed { code: Option<u16> },

    #[error("protocol violation: {0}")]
    Protocol(String),

    #[error("decision error: {0}")]
    Decision(String),
}

pub type Result<T, E = BotError> = std::result::Result<T, E>;
