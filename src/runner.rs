//! Supervisor: lobby session lifecycle + sequential match execution.
//!
//! Topology (one lobby connection per bot is a platform invariant):
//!
//! ```text
//! supervisor (this module)
//!   └─ lobby session
//!        ├─ reader task   — owns the lobby socket: pong heartbeat, forwards
//!        │                  every `matched` into an unbounded channel
//!        └─ match runner  — pops `matched` one at a time (sequential play)
//!                           and plays each match to completion
//! ```
//!
//! * A lobby drop while a match is in flight does not abort the match: the
//!   runner finishes, then the supervisor reconnects the lobby (any match the
//!   platform dispatched meanwhile is re-delivered as `resume` on reconnect).
//! * Fatal lobby errors (evict, auth 4001, revoke 4002) bubble up instead of
//!   being retried.
//! * With `loop_matches == false` the supervisor returns after the first
//!   completed match (mirrors the Python reference's `run.py`), whether it
//!   succeeded or failed.

use std::time::Duration;

use rand::RngExt;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::error::{BotError, Result};
use crate::lobby;
use crate::match_client;
use crate::proto::frames::Matched;
use crate::BotConfig;

/// What a lobby session ended with.
enum SessionOutcome {
    /// Non-loop mode: first match completed.
    FirstMatchDone,
    /// Non-loop mode: first match failed — fatal for the process.
    FirstMatchFailed(BotError),
    /// The lobby transport ended (reader task finished); carries the reader's
    /// own error so fatal conditions (evict/auth) are not swallowed.
    LobbyDown(BotError),
}

/// Run the bot. Returns after the first completed match unless
/// `cfg.loop_matches` is set (then it runs until a fatal error).
pub async fn run(cfg: BotConfig) -> Result<()> {
    cfg.validate()?;
    let engine = crate::Engine::load(&cfg.artifacts, cfg.deterministic)?;
    info!(
        base_url = %cfg.base_url,
        bot_id = %cfg.bot_id,
        token = %mask(&cfg.token),
        loop_mode = cfg.loop_matches,
        "bot starting"
    );

    let mut attempt: u32 = 0;
    loop {
        match lobby_session(&engine, &cfg).await {
            SessionOutcome::FirstMatchDone => return Ok(()),
            SessionOutcome::FirstMatchFailed(err) => return Err(err),
            // Fatal lobby conditions: replaced by a newer connection, bad
            // token, or revoked token (§8.1 close codes 4000/4001/4002).
            SessionOutcome::LobbyDown(err @ BotError::LobbyEvicted) => return Err(err),
            SessionOutcome::LobbyDown(err @ BotError::LobbyAuth { .. }) => return Err(err),
            // Everything else (network blips, server restarts) is retried
            // with exponential backoff against a fresh lobby connection.
            SessionOutcome::LobbyDown(err) => {
                attempt += 1;
                let delay = backoff(attempt);
                warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "lobby session failed; reconnecting"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// One lobby connection: reader task + sequential match execution.
async fn lobby_session(engine: &crate::Engine, cfg: &BotConfig) -> SessionOutcome {
    let connect = lobby::connect(&cfg.base_url, &cfg.bot_id, &cfg.token).await;
    let (_hello, sink, stream) = match connect {
        Ok(parts) => parts,
        Err(err) => return SessionOutcome::LobbyDown(err),
    };

    let (tx, mut rx) = mpsc::unbounded_channel::<Matched>();
    let mut reader = Some(tokio::spawn(lobby::run_read_loop(stream, sink, tx.clone())));
    // Drop our own sender so `rx.recv()` yields `None` as soon as the reader
    // task ends — that is how we notice a dead lobby between matches.
    drop(tx);

    let outcome = loop {
        let matched = match rx.recv().await {
            Some(m) => m,
            // Reader gone: surface its error (evict/auth are fatal upstream).
            None => {
                let reader_err = match reader.take() {
                    Some(handle) => handle
                        .await
                        .unwrap_or_else(|join| {
                            Err(BotError::Protocol(format!("lobby reader panicked: {join}")))
                        })
                        .err()
                        .unwrap_or(BotError::LobbyClosed),
                    None => BotError::LobbyClosed,
                };
                break SessionOutcome::LobbyDown(reader_err);
            }
        };
        if matched.resume {
            info!(match_id = %matched.match_id, "re-attaching in-flight match (resume)");
        }
        match match_client::play_match(
            engine,
            &cfg.base_url,
            &matched.gateway_ws_url,
            &matched.match_id,
            &cfg.token,
        )
        .await
        {
            Ok(end) => {
                info!(
                    match_id = %matched.match_id,
                    reason = %end.reason,
                    "match completed"
                );
                if !cfg.loop_matches {
                    break SessionOutcome::FirstMatchDone;
                }
            }
            Err(err) => {
                error!(
                    match_id = %matched.match_id,
                    error = %err,
                    "match failed"
                );
                if !cfg.loop_matches {
                    break SessionOutcome::FirstMatchFailed(err);
                }
                // Loop mode: keep the lobby alive and play the next match.
            }
        }
    };

    // Tear the reader down when the session ends without waiting for it
    // (it may be parked in a read on a live socket).
    if !matches!(outcome, SessionOutcome::LobbyDown(_)) {
        if let Some(handle) = reader.take() {
            handle.abort();
        }
    }
    outcome
}

/// Exponential backoff with jitter: 0.5 s → 30 s cap.
fn backoff(attempt: u32) -> Duration {
    let exp = 500u64.saturating_mul(1u64 << attempt.min(6));
    let cap = exp.min(30_000);
    let jitter = rand::rng().random_range(0..=250u64);
    Duration::from_millis(cap + jitter)
}

/// Mask the token for logs: keep the structural prefix + a short tail.
fn mask(token: &str) -> String {
    if token.len() <= 16 {
        "cz_extbot_****".into()
    } else {
        format!("{}…{}", &token[..12], &token[token.len() - 4..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        assert!(backoff(0) < Duration::from_millis(1000));
        assert!(backoff(1) < Duration::from_millis(1500));
        assert!(backoff(5) >= Duration::from_millis(15_000));
        assert!(backoff(10) <= Duration::from_millis(31_000));
    }

    #[test]
    fn token_masking() {
        assert_eq!(mask("cz_extbot_short"), "cz_extbot_****");
        let masked = mask("cz_extbot_0123456789abcdef0123456789abcdef");
        assert!(masked.starts_with("cz_extbot_01"));
        assert!(masked.ends_with("cdef"));
    }
}
