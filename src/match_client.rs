//! Match data-plane WebSocket client (the game loop).
//!
//! Implements the match half of the External-API path (protocol doc §5–§6):
//!
//! 1. dial `gateway_ws_url` with `Sec-WebSocket-Protocol:
//!    chipzen-bot-token, <token>` (token never travels on the query string)
//! 2. Layer-1 handshake: `authenticate` → server `hello` → client `hello`
//!    (the authenticate frame MUST be first or the executor stalls, §6.1)
//! 3. game loop: answer `ping`, decide on `turn_request` (echoing
//!    `request_id`), retry safely on `action_rejected`, stop on `match_end`.
//!    Unknown frame types are silently ignored (forward compat).
//!
//! Mid-match drops are recovered by re-dialing the same gateway URL and
//! re-running the handshake (§8.6): the executor swaps the socket and
//! re-emits the in-flight `turn_request`. The per-match budget is 3
//! reconnects; the turn clock keeps running while we are down.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite, WebSocketStream};
use tracing::{debug, info, warn};

use crate::adapter::TurnContext;
use crate::error::{BotError, Result};
use crate::proto::frames::{
    client_hello, match_authenticate, match_pong, turn_action, Decision, GameConfig, MatchEnd,
    ServerFrame,
};
use crate::strategy::Engine;

type WsStream = WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Reconnect budget per match (transport spec §15).
const MAX_MATCH_RECONNECTS: u32 = 3;

/// Play one match end-to-end. Returns the `match_end` payload.
pub async fn play_match(
    engine: &Engine,
    base_url: &str,
    gateway_ws_url: &str,
    match_id: &str,
    token: &str,
) -> Result<MatchEnd> {
    let url = resolve_gateway_url(base_url, gateway_ws_url);
    let mut reconnects = 0u32;
    loop {
        match play_once(engine, &url, match_id, token).await {
            Ok(end) => return Ok(end),
            Err(err @ BotError::MatchClosed { .. }) => {
                reconnects += 1;
                if reconnects > MAX_MATCH_RECONNECTS {
                    return Err(err);
                }
                warn!(
                    attempt = reconnects,
                    budget = MAX_MATCH_RECONNECTS,
                    "match socket dropped; re-dialing gateway (§8.6)"
                );
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(err) => return Err(err),
        }
    }
}

/// One full connection attempt: handshake + game loop.
async fn play_once(engine: &Engine, url: &str, match_id: &str, token: &str) -> Result<MatchEnd> {
    let ws = connect_gateway(url, token).await?;
    let (mut sink, mut stream) = ws.split();

    // ---- Layer-1 handshake ----
    sink.send(tungstenite::Message::text(
        match_authenticate(match_id, token).to_string(),
    ))
    .await?;
    let server_hello = loop {
        let msg = match tokio::time::timeout(Duration::from_secs(10), stream.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(_)) | None) => return Err(BotError::MatchClosed { code: None }),
            Err(_) => return Err(BotError::Protocol("match: handshake timed out".into())),
        };
        let text = crate::lobby::decode_text(msg)?;
        // Tolerate benign pre-hello frames; anything else is a violation.
        match ServerFrame::parse(&text)? {
            ServerFrame::Hello(h) => break h,
            ServerFrame::Ping | ServerFrame::Other { .. } => continue,
            other => {
                return Err(BotError::Protocol(format!(
                    "match: expected server hello, got {other:?}"
                )))
            }
        }
    };
    if server_hello.game_type.as_deref() != Some("poker") {
        warn!(game_type = ?server_hello.game_type, "match: unexpected game_type");
    }
    if server_hello.capabilities.iter().any(|c| c == "reconnect") {
        debug!("match: executor advertises reconnect capability");
    }
    sink.send(tungstenite::Message::text(
        client_hello(match_id).to_string(),
    ))
    .await?;
    info!(
        version = ?server_hello.selected_version,
        match_id = %match_id,
        "match: handshake complete"
    );

    // ---- Game loop ----
    let mut config: Option<GameConfig> = None;

    loop {
        // Turn clocks are 30 s on agent surfaces (and faster elsewhere); a
        // 90 s read timeout detects dead sockets without racing real turns.
        let msg = match tokio::time::timeout(Duration::from_secs(90), stream.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(_)) | None) => return Err(BotError::MatchClosed { code: None }),
            Err(_) => {
                return Err(BotError::MatchClosed { code: None });
            }
        };
        let text = match crate::lobby::decode_text(msg) {
            Ok(t) => t,
            Err(BotError::MatchClosed { code }) => {
                return Err(BotError::MatchClosed { code });
            }
            Err(e) => return Err(e),
        };

        match ServerFrame::parse(&text)? {
            ServerFrame::Ping => {
                debug!("match: ping → pong");
                sink.send(tungstenite::Message::text(match_pong(match_id).to_string()))
                    .await?;
            }
            ServerFrame::MatchStart { config: cfg, .. } => {
                info!(
                    variant = %cfg.variant,
                    sb = cfg.small_blind,
                    bb = cfg.big_blind,
                    stack = cfg.starting_stack,
                    players = cfg.num_players,
                    "match: game config"
                );
                config = Some(cfg);
            }
            ServerFrame::RoundStart { state, .. } => {
                debug!(
                    hand = state.hand_number,
                    dealer = state.dealer_seat,
                    hole = ?state.your_hole_cards,
                    "match: round_start"
                );
            }
            ServerFrame::TurnRequest(req) => {
                let cfg = config
                    .clone()
                    .ok_or_else(|| BotError::Protocol("turn_request before match_start".into()))?;
                let decided_at = std::time::Instant::now();
                let decision = decide_safely(engine, &req.state, &req.valid_actions, &cfg);
                tracing::info!(
                    hand = req.state.hand_number,
                    phase = %req.state.phase,
                    pot = req.state.pot,
                    to_call = req.state.to_call,
                    action = %decision.action,
                    amount = decision.amount,
                    elapsed_ms = decided_at.elapsed().as_millis() as u64,
                    remaining_ms = req.remaining_ms,
                    "match: decided"
                );
                sink.send(tungstenite::Message::text(
                    turn_action(match_id, &req.request_id, &decision).to_string(),
                ))
                .await?;
            }
            ServerFrame::ActionRejected(rej) => {
                // Retry with the SAME request_id using a guaranteed-legal
                // fallback (§6.2.1): check if legal else fold.
                let fallback = safe_fallback(&rej.valid_actions);
                warn!(
                    request_id = %rej.request_id,
                    reason = %rej.reason,
                    action = %fallback.action,
                    "match: action rejected; retrying with fallback"
                );
                sink.send(tungstenite::Message::text(
                    turn_action(match_id, &rej.request_id, &fallback).to_string(),
                ))
                .await?;
            }
            ServerFrame::MatchEnd(end) => {
                info!(reason = %end.reason, "match: ended");
                return Ok(end);
            }
            ServerFrame::Error { code, message } => {
                warn!(code = %code, message = %message, "match: error frame");
            }
            ServerFrame::Other { type_name, .. } => {
                debug!(frame = %type_name, "match: ignoring unknown frame");
            }
            other => {
                debug!(frame = ?other, "match: observed frame");
            }
        }
    }
}

/// Open the gateway WS offering the token via `Sec-WebSocket-Protocol`.
async fn connect_gateway(url: &str, token: &str) -> Result<WsStream> {
    use tungstenite::client::IntoClientRequest;
    use tungstenite::http::header::HeaderValue;

    // A plain URL gives us a fully-formed upgrade request; we only add the
    // token subprotocol header on top (CZ issue 2932 — never a query string).
    let mut request = url.into_client_request()?;
    let subprotocol = format!("{}, {}", crate::proto::frames::BOT_TOKEN_SUBPROTOCOL, token);
    request.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        HeaderValue::from_str(&subprotocol)
            .map_err(|e| BotError::Config(format!("token not header-safe: {e}")))?,
    );
    let (ws, _resp) = connect_async(request).await?;
    Ok(ws)
}

/// Resolve `gateway_ws_url` (a path) against the lobby origin.
pub fn resolve_gateway_url(base_url: &str, gateway_ws_path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if gateway_ws_path.starts_with("ws://") || gateway_ws_path.starts_with("wss://") {
        gateway_ws_path.to_string()
    } else {
        format!("{base}{gateway_ws_path}")
    }
}

/// Decide with the engine, falling back to a guaranteed-legal action if the
/// adapter or strategy hits an unexpected state (a panicked hand must never
/// forfeit the match).
fn decide_safely(
    engine: &Engine,
    state: &crate::proto::frames::TurnState,
    valid_actions: &[String],
    config: &GameConfig,
) -> Decision {
    let fallback = || {
        if state.to_call <= 0 && valid_actions.iter().any(|a| a == "check") {
            Decision::check()
        } else if valid_actions.iter().any(|a| a == "call") {
            Decision::call()
        } else if valid_actions.iter().any(|a| a == "check") {
            Decision::check()
        } else {
            Decision::fold()
        }
    };
    match TurnContext::from_turn_state(state, valid_actions, config) {
        Ok(ctx) => {
            let d = engine.decide(&ctx);
            // Belt-and-suspenders: hard-verify against the wire's own
            // valid_actions (the engine validated against our parse).
            let ok = match d.action.as_str() {
                "raise" => {
                    valid_actions.iter().any(|a| a == "raise")
                        && d.amount
                            .map(|a| (ctx.min_raise..=ctx.max_raise).contains(&a))
                            .unwrap_or(false)
                }
                a => valid_actions.iter().any(|v| v == a),
            };
            if ok {
                d
            } else {
                warn!(action = %d.action, "decision failed wire validation; using safe fallback");
                fallback()
            }
        }
        Err(e) => {
            warn!(error = %e, "failed to build TurnContext; using safe fallback");
            fallback()
        }
    }
}

/// Guaranteed-legal fallback per the reference client: check if legal,
/// else fold (the server auto-action guarantee).
fn safe_fallback(valid_actions: &[String]) -> Decision {
    if valid_actions.iter().any(|a| a == "check") {
        Decision::check()
    } else if valid_actions.iter().any(|a| a == "fold") {
        Decision::fold()
    } else {
        Decision::check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_url_resolution() {
        assert_eq!(
            resolve_gateway_url("wss://staging.chipzen.ai", "/ws/external/match/m1/p1"),
            "wss://staging.chipzen.ai/ws/external/match/m1/p1"
        );
        assert_eq!(
            resolve_gateway_url("wss://host/", "wss://other/match"),
            "wss://other/match"
        );
    }

    #[test]
    fn safe_fallback_prefers_check_then_fold() {
        assert_eq!(
            safe_fallback(&["fold".into(), "check".into()]),
            Decision::check()
        );
        assert_eq!(
            safe_fallback(&["fold".into(), "call".into()]),
            Decision::fold()
        );
        assert_eq!(safe_fallback(&["call".into()]), Decision::check());
    }
}
