//! Chipzen External-API wire frames (Layer-1 transport + Layer-2 poker payloads).
//!
//! Mirrors `docs/EXTERNAL-API-BOT-PROTOCOL.md` and
//! `docs/protocol/POKER-GAME-STATE-PROTOCOL.md` of the chipzen-sdk, plus the
//! behaviour of the reference Python client (`examples/external-api-bot`).
//!
//! Parsing strategy: every inbound frame arrives as `serde_json::Value`; the
//! `"type"` field is dispatched manually so that *unknown frame types are
//! silently ignored* (protocol forward-compat requirement, §6.2 of the
//! external-API doc). Known types are deserialized into strongly typed structs
//! with liberal `#[serde(default)]` so new additive fields never break us.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{BotError, Result};

pub const PROTOCOL_VERSIONS: [&str; 1] = ["1.0"];
pub const CLIENT_NAME: &str = "pkr-chipzen-bot";
pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Sentinel subprotocol that marks the `cz_extbot_` token in the
/// `Sec-WebSocket-Protocol` header (CZ issue 2932).
pub const BOT_TOKEN_SUBPROTOCOL: &str = "chipzen-bot-token";

// ---------------------------------------------------------------------------
// Outbound (bot → server) frames
// ---------------------------------------------------------------------------

/// Lobby authenticate frame — must be the FIRST frame on the lobby socket.
pub fn lobby_authenticate(token: &str) -> Value {
    serde_json::json!({ "type": "authenticate", "token": token })
}

/// Match-gateway authenticate frame — must be the FIRST frame on the match
/// socket. The executor drains the credential (the gateway's internal JWT is
/// authoritative) but the frame itself is mandatory, or the handshake stalls.
pub fn match_authenticate(match_id: &str, token: &str) -> Value {
    serde_json::json!({ "type": "authenticate", "match_id": match_id, "token": token })
}

/// Client hello — second leg of the Layer-1 match handshake.
pub fn client_hello(match_id: &str) -> Value {
    serde_json::json!({
        "type": "hello",
        "match_id": match_id,
        "supported_versions": PROTOCOL_VERSIONS,
        "client_name": CLIENT_NAME,
        "client_version": CLIENT_VERSION,
    })
}

/// Lobby heartbeat reply (no match id on the lobby socket).
pub fn lobby_pong() -> Value {
    serde_json::json!({ "type": "pong" })
}

/// Match heartbeat reply.
pub fn match_pong(match_id: &str) -> Value {
    serde_json::json!({ "type": "pong", "match_id": match_id })
}

/// The action reply. `amount` is included only for `raise`.
pub fn turn_action(match_id: &str, request_id: &str, decision: &Decision) -> Value {
    let mut params = Map::new();
    if let Some(amount) = decision.amount {
        params.insert("amount".into(), Value::from(amount));
    }
    serde_json::json!({
        "type": "turn_action",
        "match_id": match_id,
        "request_id": request_id,
        "action": decision.action,
        "params": params,
    })
}

/// A chosen action, independent of the wire envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub action: String,
    /// Total bet size for `raise` (`min_raise <= amount <= max_raise`).
    pub amount: Option<i64>,
}

impl Decision {
    pub fn fold() -> Self {
        Decision { action: "fold".into(), amount: None }
    }
    pub fn check() -> Self {
        Decision { action: "check".into(), amount: None }
    }
    pub fn call() -> Self {
        Decision { action: "call".into(), amount: None }
    }
    pub fn raise(amount: i64) -> Self {
        Decision { action: "raise".into(), amount: Some(amount) }
    }
}

// ---------------------------------------------------------------------------
// Inbound (server → bot) frames
// ---------------------------------------------------------------------------

/// Server hello (both lobby and match sockets).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ServerHello {
    #[serde(default)]
    pub match_id: String,
    #[serde(default)]
    pub selected_version: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub game_type: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// `matched` notification delivered on the lobby socket.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Matched {
    #[serde(default)]
    pub match_id: String,
    #[serde(default)]
    pub participant_id: String,
    #[serde(default)]
    pub gateway_ws_url: String,
    #[serde(default)]
    pub rated: bool,
    #[serde(default)]
    pub decision_timeout_ms: Option<u64>,
    /// `true` when this is a re-attach of an in-flight match (§8.6).
    #[serde(default)]
    pub resume: bool,
}

/// `game_config` from `match_start` (Layer-2 §3.1).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct GameConfig {
    #[serde(default = "default_variant")]
    pub variant: String,
    pub starting_stack: i64,
    pub small_blind: i64,
    pub big_blind: i64,
    #[serde(default)]
    pub ante: i64,
    #[serde(default = "default_num_players")]
    pub num_players: usize,
}

fn default_variant() -> String {
    "nlhe".into()
}
fn default_num_players() -> usize {
    2
}

/// One entry of `action_history` (Layer-2 §3.4).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ActionEntry {
    pub seat: i64,
    pub action: String,
    #[serde(default)]
    pub amount: i64,
    #[serde(default)]
    pub phase: String,
    #[serde(default)]
    pub is_timeout: bool,
}

/// `turn_request.state` (Layer-2 §3.3) — the core decision payload.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnState {
    #[serde(default)]
    pub hand_number: i64,
    pub phase: String,
    #[serde(default)]
    pub board: Vec<String>,
    pub your_hole_cards: Vec<String>,
    pub pot: i64,
    pub your_stack: i64,
    #[serde(default)]
    pub opponent_stacks: Vec<i64>,
    pub to_call: i64,
    #[serde(default)]
    pub min_raise: i64,
    #[serde(default)]
    pub max_raise: i64,
    #[serde(default)]
    pub action_history: Vec<ActionEntry>,
}

/// `turn_request` frame (Layer-1 envelope + Layer-2 state).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnRequest {
    #[serde(default)]
    pub match_id: String,
    #[serde(default)]
    pub request_id: String,
    pub state: TurnState,
    #[serde(default)]
    pub valid_actions: Vec<String>,
    #[serde(default)]
    pub remaining_ms: Option<u64>,
}

/// `action_rejected` — our previous action was illegal; retry with the same
/// `request_id` within `remaining_ms` (§6.2.1).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ActionRejected {
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub valid_actions: Vec<String>,
    #[serde(default)]
    pub remaining_ms: Option<u64>,
}

/// `match_end` payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MatchEnd {
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub results: Value,
}

/// `round_start.state` (Layer-2 §3.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RoundStartState {
    #[serde(default)]
    pub hand_number: i64,
    #[serde(default)]
    pub dealer_seat: i64,
    #[serde(default)]
    pub your_hole_cards: Vec<String>,
    #[serde(default)]
    pub stacks: Vec<i64>,
}

/// A generically-parsed inbound frame.
#[derive(Debug, Clone)]
pub enum ServerFrame {
    Hello(ServerHello),
    Ping,
    Matched(Matched),
    Evict,
    MatchStart { match_id: String, config: GameConfig },
    RoundStart { match_id: String, state: RoundStartState },
    TurnRequest(TurnRequest),
    ActionRejected(ActionRejected),
    MatchEnd(MatchEnd),
    Error { code: String, message: String },
    /// Any other type — MUST be ignored (forward compat).
    Other { type_name: String, value: Value },
}

impl ServerFrame {
    /// Parse one JSON text frame into a typed frame.
    pub fn parse(raw: &str) -> Result<ServerFrame> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|e| BotError::Protocol(format!("bad json frame: {e}; raw={raw:.200}")))?;
        let type_name = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let match_id = || {
            value
                .get("match_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let frame = match type_name.as_str() {
            "hello" => ServerFrame::Hello(serde_json::from_value(value)?),
            "ping" => ServerFrame::Ping,
            "matched" => ServerFrame::Matched(serde_json::from_value(value)?),
            "evict" => ServerFrame::Evict,
            "match_start" => {
                let config = value
                    .get("game_config")
                    .cloned()
                    .map(serde_json::from_value::<GameConfig>)
                    .transpose()?
                    .ok_or_else(|| {
                        BotError::Protocol("match_start missing game_config".into())
                    })?;
                ServerFrame::MatchStart { match_id: match_id(), config }
            }
            "round_start" => {
                let state = value
                    .get("state")
                    .cloned()
                    .map(serde_json::from_value::<RoundStartState>)
                    .transpose()?
                    .unwrap_or(RoundStartState {
                        hand_number: 0,
                        dealer_seat: 0,
                        your_hole_cards: vec![],
                        stacks: vec![],
                    });
                ServerFrame::RoundStart { match_id: match_id(), state }
            }
            "turn_request" => {
                let req: TurnRequest = serde_json::from_value(value)?;
                ServerFrame::TurnRequest(req)
            }
            "action_rejected" => {
                ServerFrame::ActionRejected(serde_json::from_value(value)?)
            }
            "match_end" => ServerFrame::MatchEnd(serde_json::from_value(value)?),
            "error" => ServerFrame::Error {
                code: value
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                message: value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            other => ServerFrame::Other { type_name: other.to_string(), value },
        };
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TURN_REQUEST_EXAMPLE: &str = r#"{
      "type": "turn_request",
      "match_id": "m_abc123",
      "seq": 7,
      "request_id": "req-42",
      "remaining_ms": 29500,
      "valid_actions": ["fold", "call", "raise"],
      "state": {
        "hand_number": 1,
        "phase": "flop",
        "board": ["Ts", "7h", "2d"],
        "your_hole_cards": ["Ah", "Kd"],
        "pot": 80,
        "your_stack": 970,
        "opponent_stacks": [950],
        "to_call": 20,
        "min_raise": 40,
        "max_raise": 970,
        "action_history": [
          {"seat": 0, "action": "post_small_blind", "amount": 5, "phase": "preflop", "is_timeout": false},
          {"seat": 1, "action": "post_big_blind", "amount": 10, "phase": "preflop", "is_timeout": false},
          {"seat": 0, "action": "raise", "amount": 30, "phase": "preflop", "is_timeout": false},
          {"seat": 1, "action": "call", "amount": 30, "phase": "preflop", "is_timeout": false},
          {"seat": 1, "action": "raise", "amount": 20, "phase": "flop", "is_timeout": false}
        ]
      }
    }"#;

    #[test]
    fn parses_documented_turn_request() {
        let f = ServerFrame::parse(TURN_REQUEST_EXAMPLE).unwrap();
        let req = match f {
            ServerFrame::TurnRequest(r) => r,
            other => panic!("expected turn_request, got {other:?}"),
        };
        assert_eq!(req.request_id, "req-42");
        assert_eq!(req.state.phase, "flop");
        assert_eq!(req.state.pot, 80);
        assert_eq!(req.state.to_call, 20);
        assert_eq!(req.state.min_raise, 40);
        assert_eq!(req.state.max_raise, 970);
        assert_eq!(req.state.board.len(), 3);
        assert_eq!(req.state.your_hole_cards, vec!["Ah", "Kd"]);
        assert_eq!(req.valid_actions, vec!["fold", "call", "raise"]);
        assert_eq!(req.state.action_history.len(), 5);
        assert_eq!(req.state.action_history[2].action, "raise");
        assert_eq!(req.state.action_history[2].amount, 30);
    }

    #[test]
    fn parses_matched_with_resume_and_timeout() {
        let raw = r#"{"type":"matched","match_id":"m1","participant_id":"p1",
                      "gateway_ws_url":"/ws/external/match/m1/p1","rated":true,
                      "decision_timeout_ms":30000,"resume":true}"#;
        match ServerFrame::parse(raw).unwrap() {
            ServerFrame::Matched(m) => {
                assert_eq!(m.match_id, "m1");
                assert_eq!(m.gateway_ws_url, "/ws/external/match/m1/p1");
                assert!(m.rated);
                assert!(m.resume);
                assert_eq!(m.decision_timeout_ms, Some(30_000));
            }
            other => panic!("expected matched, got {other:?}"),
        }
    }

    #[test]
    fn unknown_frame_types_are_forward_compatible() {
        let raw = r#"{"type":"session_control","frobnicate":true}"#;
        match ServerFrame::parse(raw).unwrap() {
            ServerFrame::Other { type_name, .. } => assert_eq!(type_name, "session_control"),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn parses_match_start_game_config() {
        let raw = r#"{"type":"match_start","match_id":"m1","seq":1,
            "game_config":{"variant":"nlhe","starting_stack":1000,
            "small_blind":5,"big_blind":10,"ante":0,"num_players":2}}"#;
        match ServerFrame::parse(raw).unwrap() {
            ServerFrame::MatchStart { match_id, config } => {
                assert_eq!(match_id, "m1");
                assert_eq!(config.starting_stack, 1000);
                assert_eq!(config.small_blind, 5);
                assert_eq!(config.big_blind, 10);
                assert_eq!(config.ante, 0);
                assert_eq!(config.num_players, 2);
            }
            other => panic!("expected match_start, got {other:?}"),
        }
    }

    #[test]
    fn turn_action_serialization_matches_reference_shape() {
        // raise to 60
        let d = Decision::raise(60);
        let v = turn_action("m1", "req-1", &d);
        assert_eq!(
            v,
            serde_json::json!({
                "type":"turn_action","match_id":"m1","request_id":"req-1",
                "action":"raise","params":{"amount":60}
            })
        );
        // check — params is an empty object
        let v = turn_action("m1", "req-2", &Decision::check());
        assert_eq!(v["params"], serde_json::json!({}));
        assert_eq!(v["action"], "check");
    }

    #[test]
    fn action_rejected_parses() {
        let raw = r#"{"type":"action_rejected","match_id":"m1","request_id":"r9",
                      "reason":"not_your_turn","valid_actions":["check","fold"],
                      "remaining_ms":21000}"#;
        match ServerFrame::parse(raw).unwrap() {
            ServerFrame::ActionRejected(r) => {
                assert_eq!(r.request_id, "r9");
                assert_eq!(r.reason, "not_your_turn");
                assert_eq!(r.valid_actions, vec!["check", "fold"]);
            }
            other => panic!("expected action_rejected, got {other:?}"),
        }
    }
}
