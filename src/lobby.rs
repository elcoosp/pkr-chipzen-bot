//! Lobby WebSocket client.
//!
//! Implements the lobby half of the External-API path (protocol doc §3–§4):
//!
//! 1. open `wss://<host>/ws/external/bot/{bot_id}`
//! 2. send `{"type":"authenticate","token":…}` as the first frame
//! 3. receive the server `hello` (`endpoint == "lobby"`) — no client hello back
//! 4. answer application-level `ping` with `pong` (server pings every 15 s,
//!    expects the pong within 5 s)
//! 5. yield every `matched` notification (fresh or `resume` re-attach)
//!
//! A single lobby connection per bot is a platform invariant — opening a
//! second one evicts the first (close 4000), which this client treats as a
//! fatal configuration error (`BotError::LobbyEvicted`).

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite, WebSocketStream};
use tracing::{debug, info, warn};

use crate::error::{BotError, Result};
use crate::proto::frames::{lobby_authenticate, lobby_pong, Matched, ServerFrame, ServerHello};

/// Anything the supervisor needs to know about a live lobby connection.
pub struct Lobby {
    /// Sends text frames to the lobby (pong replies).
    pub sink: futures_util::stream::SplitSink<
        WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        tungstenite::Message,
    >,
}

/// Connect to the lobby, run the authenticate handshake and return the
/// server hello plus a stream/split pair for further IO.
pub async fn connect(
    base_url: &str,
    bot_id: &str,
    token: &str,
) -> Result<(
    ServerHello,
    futures_util::stream::SplitSink<
        WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        tungstenite::Message,
    >,
    futures_util::stream::SplitStream<
        WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    >,
)> {
    let url = format!(
        "{}/ws/external/bot/{bot_id}",
        base_url.trim_end_matches('/')
    );
    info!(url = %url, "lobby: connecting");
    let (ws, _resp) = connect_async(url.as_str()).await?;
    let (mut sink, mut stream) = ws.split();

    // First frame must be authenticate.
    sink.send(tungstenite::Message::text(
        lobby_authenticate(token).to_string(),
    ))
    .await?;

    // Wait for the server hello (or an early close).
    let hello = loop {
        match tokio::time::timeout(Duration::from_secs(15), stream.next()).await {
            Ok(Some(Ok(msg))) => {
                let text = decode_text(msg).map_err(into_lobby_error)?;
                match ServerFrame::parse(&text)? {
                    ServerFrame::Hello(h) => break h,
                    ServerFrame::Error { code, message } => {
                        return Err(BotError::Protocol(format!(
                            "lobby error [{code}] {message}"
                        )));
                    }
                    _ => debug!("lobby: ignoring pre-hello frame"),
                }
            }
            Ok(Some(Err(e))) => return Err(e.into()),
            Ok(None) => return Err(BotError::LobbyClosed),
            Err(_) => {
                return Err(BotError::Protocol(
                    "lobby: timed out waiting for server hello".into(),
                ));
            }
        }
    };
    if hello.endpoint.as_deref() != Some("lobby") {
        warn!(endpoint = ?hello.endpoint, "lobby: unexpected hello endpoint");
    }
    info!(version = ?hello.selected_version, "lobby: connected");
    Ok((hello, sink, stream))
}

/// Drive the lobby read loop, forwarding `matched` notifications into `tx`.
///
/// Heartbeat: every server `ping` is answered with `pong` through `sink`.
/// Returns when the lobby socket closes (the supervisor decides whether to
/// reconnect). Fatal conditions (evict, close code 4001/4002) are surfaced as
/// errors.
pub async fn run_read_loop(
    mut stream: futures_util::stream::SplitStream<
        WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    >,
    mut sink: futures_util::stream::SplitSink<
        WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        tungstenite::Message,
    >,
    tx: mpsc::UnboundedSender<Matched>,
) -> Result<()> {
    while let Some(msg) = stream.next().await {
        let text = decode_text(msg?).map_err(into_lobby_error)?;
        match ServerFrame::parse(&text)? {
            ServerFrame::Ping => {
                debug!("lobby: ping → pong");
                sink.send(tungstenite::Message::text(lobby_pong().to_string()))
                    .await?;
            }
            ServerFrame::Matched(m) => {
                info!(
                    match_id = %m.match_id,
                    participant = %m.participant_id,
                    rated = m.rated,
                    resume = m.resume,
                    "lobby: matched"
                );
                // Unbounded: the platform's concurrent-match cap is small; a
                // burst of notifications is played off sequentially.
                let _ = tx.send(m);
            }
            ServerFrame::Evict => return Err(BotError::LobbyEvicted),
            ServerFrame::Error { code, message } => {
                warn!(code = %code, message = %message, "lobby: error frame");
            }
            ServerFrame::Hello(h) => {
                debug!(endpoint = ?h.endpoint, "lobby: duplicate hello");
            }
            ServerFrame::Other { type_name, .. } => {
                debug!(frame = %type_name, "lobby: ignoring unknown frame");
            }
            other => debug!(frame = ?other, "lobby: unexpected frame on lobby socket"),
        }
    }
    Ok(())
}

/// Decode a tungstenite message into a JSON text payload.
pub(crate) fn decode_text(msg: tungstenite::Message) -> Result<String> {
    match msg {
        tungstenite::Message::Text(s) => Ok(s.to_string()),
        tungstenite::Message::Binary(b) => String::from_utf8(b)
            .map_err(|e| BotError::Protocol(format!("binary frame is not utf-8: {e}"))),
        tungstenite::Message::Close(close) => Err(BotError::MatchClosed {
            code: close.map(|c| u16::from(c.code)),
        }),
        tungstenite::Message::Ping(_) | tungstenite::Message::Pong(_) => {
            // Protocol-level ping/pong is handled by tungstenite itself.
            Err(BotError::Protocol(
                "unexpected protocol-level ping/pong".into(),
            ))
        }
        tungstenite::Message::Frame(_) => Err(BotError::Protocol("unexpected raw frame".into())),
    }
}

/// Interpret a WS close frame received on the **lobby** socket
/// (external-API doc §8.1).
fn into_lobby_error(err: BotError) -> BotError {
    let BotError::MatchClosed { code } = err else {
        return err;
    };
    match code {
        // 4000: heartbeat failed OR replaced by a newer connection. The
        // explicit `evict` frame is the primary eviction signal; a bare 4000
        // close is almost always the eviction close that follows it.
        Some(4000) => BotError::LobbyEvicted,
        // 4001: auth failed (bad/foreign bot token). 4002: token revoked.
        Some(c @ (4001 | 4002)) => BotError::LobbyAuth { code: c },
        other => BotError::MatchClosed { code: other },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lobby_authenticate_is_first_frame_shape() {
        let v = lobby_authenticate("cz_extbot_x");
        assert_eq!(v["type"], "authenticate");
        assert_eq!(v["token"], "cz_extbot_x");
    }

    #[test]
    fn lobby_pong_has_no_match_id() {
        let v = lobby_pong();
        assert_eq!(v["type"], "pong");
        assert!(v.get("match_id").is_none());
    }
}
