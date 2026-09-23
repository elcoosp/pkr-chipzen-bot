//! End-to-end integration test: a mock chipzen platform (lobby + match
//! gateway) drives the real bot through one full match:
//!
//! lobby auth → hello → ping/pong → matched →
//! match handshake (token via `Sec-WebSocket-Protocol`) → match_start →
//! 4 turn_requests exercising [blueprint check-hit, call-hit, shove-hit,
//! miss→fallback] → action_rejected → guaranteed fallback → match_end.
//!
//! All synthetic artifacts are deterministic (see `common`), so every
//! expected action is exact. The wire payloads are serialized from the very
//! same `TurnState` structs used to compute the blueprint keys — zero drift.

// The test binary doesn't inherit the lib's lint allows.
#![allow(clippy::result_large_err)]

mod common;

use std::sync::{Arc, Mutex};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::http::{header::HeaderValue, Request, Response};
use tokio_tungstenite::tungstenite::Message;

use pkr_chipzen_bot::adapter::TurnContext;
use pkr_chipzen_bot::proto::frames::{ActionEntry, GameConfig, TurnState};
use pkr_chipzen_bot::strategy::ArtifactPaths;
use pkr_chipzen_bot::BotConfig;

const TOKEN: &str = "cz_extbot_testtoken12345678901234567890";
const MATCH_ID: &str = "m_abc123";
const PARTICIPANT: &str = "p_seat0";

/// One scripted turn: the state the bot will receive, the valid actions,
/// and the exact action (+amount) the deterministic engine must produce.
struct Scenario {
    state: TurnState,
    valid: Vec<String>,
    expected_action: &'static str,
    expected_amount: Option<i64>,
}

fn game_config() -> GameConfig {
    GameConfig {
        variant: "nlhe".into(),
        starting_stack: 1000,
        small_blind: 5,
        big_blind: 10,
        ante: 0,
        num_players: 2,
    }
}

fn entry(seat: i64, action: &str, amount: i64, phase: &str) -> ActionEntry {
    ActionEntry {
        seat,
        action: action.into(),
        amount,
        phase: phase.into(),
        is_timeout: false,
    }
}

fn blinds() -> Vec<ActionEntry> {
    vec![
        entry(0, "post_small_blind", 5, "preflop"),
        entry(1, "post_big_blind", 10, "preflop"),
    ]
}

#[allow(clippy::too_many_arguments)]
fn state(
    phase: &str,
    board: Vec<&str>,
    to_call: i64,
    min_raise: i64,
    max_raise: i64,
    your_stack: i64,
    pot: i64,
    history: Vec<ActionEntry>,
) -> TurnState {
    TurnState {
        hand_number: 1,
        phase: phase.into(),
        board: board.into_iter().map(String::from).collect(),
        your_hole_cards: vec!["Ah".into(), "Kd".into()],
        pot,
        your_stack,
        opponent_stacks: vec![your_stack],
        to_call,
        min_raise,
        max_raise,
        action_history: history,
    }
}

/// Read the next JSON text frame, skipping protocol-level noise.
async fn next_json<S>(stream: &mut S) -> Option<serde_json::Value>
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>
        + std::marker::Unpin,
{
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Text(t)) => return Some(serde_json::from_str(&t).expect("valid json")),
            Ok(Message::Close(c)) => {
                tracing::debug!(?c, "mock: client closed");
                return None;
            }
            Ok(_) => continue,
            Err(e) => panic!("mock: ws read error: {e}"),
        }
    }
    None
}

async fn send_json<S>(sink: &mut S, v: &serde_json::Value)
where
    S: futures_util::Sink<Message> + std::marker::Unpin,
    S::Error: std::fmt::Debug,
{
    sink.send(Message::text(v.to_string()))
        .await
        .expect("mock send");
}

#[tokio::test]
async fn full_match_against_mock_platform() {
    let artifact_dir = tempfile::tempdir().expect("artifact dir");
    let cfg = game_config();

    // ---- Build the four turn requests and the blueprint keys ----

    // r1: preflop, free check, blueprint hit on slot 1 → check.
    let s1 = state("preflop", vec![], 0, 20, 995, 995, 15, blinds());
    let valid1 = vec!["check".to_string(), "fold".to_string()];
    let ctx1 = TurnContext::from_turn_state(&s1, &valid1, &cfg).expect("ctx1");

    // r2: flop facing 20 into 80, blueprint hit on slot 1 → call.
    let mut h2 = blinds();
    h2.push(entry(0, "raise", 30, "preflop"));
    h2.push(entry(1, "call", 30, "preflop"));
    h2.push(entry(1, "raise", 20, "flop"));
    let s2 = state("flop", vec!["Ts", "7h", "2d"], 20, 40, 970, 970, 80, h2);
    let valid2 = vec!["fold".to_string(), "call".to_string(), "raise".to_string()];
    let ctx2 = TurnContext::from_turn_state(&s2, &valid2, &cfg).expect("ctx2");

    // r3: turn, free, blueprint hit on slot 5 → shove (raise max_raise 400).
    let mut h3 = blinds();
    h3.push(entry(0, "raise", 30, "preflop"));
    h3.push(entry(1, "call", 30, "preflop"));
    h3.push(entry(1, "raise", 20, "flop"));
    h3.push(entry(0, "call", 20, "flop"));
    h3.push(entry(1, "check", 0, "turn"));
    let s3 = state(
        "turn",
        vec!["Ts", "7h", "2d", "9c"],
        0,
        100,
        400,
        400,
        80,
        h3,
    );
    let valid3 = vec!["check".to_string(), "raise".to_string()];
    let ctx3 = TurnContext::from_turn_state(&s3, &valid3, &cfg).expect("ctx3");

    // r4: river facing 100 into 500, blueprint MISS → EHS fallback → call.
    let mut h4 = blinds();
    h4.push(entry(0, "raise", 30, "preflop"));
    h4.push(entry(1, "call", 30, "preflop"));
    h4.push(entry(1, "raise", 20, "flop"));
    h4.push(entry(0, "call", 20, "flop"));
    h4.push(entry(1, "check", 0, "turn"));
    h4.push(entry(0, "check", 0, "turn"));
    h4.push(entry(1, "raise", 100, "river"));
    let s4 = state(
        "river",
        vec!["Ts", "7h", "2d", "9c", "4s"],
        100,
        200,
        400,
        300,
        500,
        h4,
    );
    let valid4 = vec!["fold".to_string(), "call".to_string()];
    let _ctx4 = TurnContext::from_turn_state(&s4, &valid4, &cfg).expect("ctx4");

    // Blueprint: hashes of r1..r3 (r4 deliberately absent). The base
    // artifacts must exist before hashes can be computed.
    common::build_base_artifacts(artifact_dir.path());
    let entries = vec![
        (
            common::synthetic_hash(artifact_dir.path(), &ctx1),
            common::cdf_all_on(1),
        ),
        (
            common::synthetic_hash(artifact_dir.path(), &ctx2),
            common::cdf_all_on(1),
        ),
        (
            common::synthetic_hash(artifact_dir.path(), &ctx3),
            common::cdf_all_on(5),
        ),
    ];
    let artifacts: ArtifactPaths = common::build_artifacts(artifact_dir.path(), &entries);

    // ---- Scripted turns ----
    let scenarios = vec![
        // r1: preflop, free check, blueprint hit on slot 1 → check.
        Scenario {
            state: s1,
            valid: valid1,
            expected_action: "check",
            expected_amount: None,
        },
        // r2: flop facing 20 into 80, blueprint hit on slot 1 → call.
        Scenario {
            state: s2,
            valid: valid2,
            expected_action: "call",
            expected_amount: None,
        },
        // r3: turn, free, blueprint hit on slot 5 → shove (raise max_raise 400).
        Scenario {
            state: s3,
            valid: valid3,
            expected_action: "raise",
            expected_amount: Some(400),
        },
        // r4: river facing 100 into 500, blueprint MISS → EHS fallback → call.
        Scenario {
            state: s4,
            valid: valid4,
            expected_action: "call",
            expected_amount: None,
        },
    ];

    // ---- Mock platform ----
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let base_url = format!("ws://{addr}");

    let server = tokio::spawn(async move {
        run_mock_platform(listener, scenarios).await;
    });

    // ---- The bot ----
    let bot_cfg = BotConfig {
        base_url,
        bot_id: "00000000-0000-0000-0000-000000000001".into(),
        token: TOKEN.into(),
        loop_matches: false,
        artifacts,
        deterministic: true,
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        pkr_chipzen_bot::run(bot_cfg),
    )
    .await
    .expect("bot must finish within the timeout");
    assert!(result.is_ok(), "bot run failed: {result:?}");

    // Surface any panic from inside the server task.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), server).await;
}

/// One lobby leg + one match leg, with inline assertions. Panics propagate
/// through the task join handle.
async fn run_mock_platform(listener: TcpListener, scenarios: Vec<Scenario>) {
    // ---------------- Lobby leg ----------------
    let (tcp, _) = listener.accept().await.expect("lobby tcp");
    let ws = accept_hdr_async(tcp, |_req: &Request<_>, resp: Response<()>| Ok(resp))
        .await
        .expect("lobby ws");
    let (mut sink, mut stream) = ws.split();

    let auth = next_json(&mut stream).await.expect("lobby auth frame");
    assert_eq!(
        auth["type"], "authenticate",
        "first lobby frame must authenticate"
    );
    assert_eq!(auth["token"], TOKEN);

    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "hello", "match_id": "bot-uuid", "seq": 1,
            "supported_versions": ["1.0"], "selected_version": "1.0",
            "server_name": "chipzen", "endpoint": "lobby"
        }),
    )
    .await;

    // Heartbeat: ping → pong.
    send_json(&mut sink, &serde_json::json!({"type": "ping"})).await;
    let pong = next_json(&mut stream).await.expect("pong");
    assert_eq!(pong["type"], "pong", "bot must answer lobby ping with pong");

    // Dispatch the match.
    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "matched",
            "match_id": MATCH_ID,
            "participant_id": PARTICIPANT,
            "gateway_ws_url": format!("/ws/external/match/{MATCH_ID}/{PARTICIPANT}"),
            "rated": false
        }),
    )
    .await;

    // Keep the lobby leg open (drain) while the match runs, in a separate
    // task; it ends when the bot drops the socket after the match.
    tokio::spawn(async move {
        while let Some(_frame) = next_json(&mut stream).await {
            // The bot stays silent on the lobby socket during a match.
        }
    });

    // ---------------- Match leg ----------------
    let (tcp, _) = listener.accept().await.expect("match tcp");
    let captured = Arc::new(Mutex::new(String::new()));
    let captured_cb = captured.clone();
    let ws = accept_hdr_async(tcp, move |req: &Request<_>, mut resp: Response<()>| {
        if let Some(p) = req.headers().get("Sec-WebSocket-Protocol") {
            *captured_cb.lock().unwrap() = p.to_str().unwrap_or_default().to_string();
        }
        // The real gateway echoes the sentinel; the client must accept it.
        resp.headers_mut().insert(
            "Sec-WebSocket-Protocol",
            HeaderValue::from_static("chipzen-bot-token"),
        );
        Ok(resp)
    })
    .await
    .expect("match ws");
    let (mut sink, mut stream) = ws.split();

    let auth = next_json(&mut stream).await.expect("match auth frame");
    assert_eq!(auth["type"], "authenticate");
    assert_eq!(auth["match_id"], MATCH_ID);

    let sub = captured.lock().unwrap().clone();
    assert!(
        sub.starts_with("chipzen-bot-token,") && sub.contains(TOKEN),
        "token must travel in Sec-WebSocket-Protocol, got {sub:?}"
    );

    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "hello", "match_id": MATCH_ID, "seq": 1,
            "supported_versions": ["1.0"], "selected_version": "1.0",
            "game_type": "poker", "capabilities": ["reconnect"]
        }),
    )
    .await;
    let hello = next_json(&mut stream).await.expect("client hello");
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["match_id"], MATCH_ID);
    assert_eq!(hello["supported_versions"], serde_json::json!(["1.0"]));

    // match_start
    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "match_start", "match_id": MATCH_ID, "seq": 2,
            "game_config": {
                "variant": "nlhe", "starting_stack": 1000,
                "small_blind": 5, "big_blind": 10, "ante": 0, "num_players": 2
            }
        }),
    )
    .await;
    // round_start
    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "round_start", "match_id": MATCH_ID, "seq": 3,
            "state": {
                "hand_number": 1, "dealer_seat": 0,
                "your_hole_cards": ["Ah", "Kd"],
                "pot": 15, "stacks": [1000, 1000], "deck_commitment": ""
            }
        }),
    )
    .await;

    // ---- Scripted turns: r1..r4 ----
    for (i, sc) in scenarios.iter().enumerate() {
        let request_id = format!("req-{}", i + 1);
        send_turn_request(
            &mut sink,
            &mut stream,
            &request_id,
            &sc.state,
            &sc.valid,
            Some((sc.expected_action, sc.expected_amount)),
        )
        .await;
    }

    // ---- action_rejected: retry same request_id with check ----
    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "action_rejected", "match_id": MATCH_ID,
            "request_id": "req-4", "reason": "unit-test",
            "valid_actions": ["check", "fold"], "remaining_ms": 20000
        }),
    )
    .await;
    let retry = next_json(&mut stream).await.expect("retry action");
    assert_eq!(retry["type"], "turn_action");
    assert_eq!(
        retry["request_id"], "req-4",
        "retry must echo the same request_id"
    );
    assert_eq!(
        retry["action"], "check",
        "rejection fallback must be check here"
    );

    // ---- match_end ----
    send_json(
        &mut sink,
        &serde_json::json!({
            "type": "match_end", "match_id": MATCH_ID, "reason": "completed",
            "results": {"winner": "nobody"}
        }),
    )
    .await;

    // Bot should close the match socket after match_end.
    let _ = sink.close().await;
}

/// Send a `turn_request` for `state`, then assert the bot's `turn_action`
/// matches `expected`.
async fn send_turn_request<Sk, St>(
    sink: &mut Sk,
    stream: &mut St,
    request_id: &str,
    state: &TurnState,
    valid_actions: &[String],
    expected: Option<(&str, Option<i64>)>,
) where
    Sk: futures_util::Sink<Message> + std::marker::Unpin,
    Sk::Error: std::fmt::Debug,
    St: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>>
        + std::marker::Unpin,
{
    let state_json = serde_json::to_value(state).expect("serialize state");
    send_json(
        sink,
        &serde_json::json!({
            "type": "turn_request",
            "match_id": MATCH_ID,
            "request_id": request_id,
            "remaining_ms": 29000,
            "valid_actions": valid_actions,
            "state": state_json
        }),
    )
    .await;
    if let Some((action, amount)) = expected {
        let reply = next_json(stream).await.expect("turn_action reply");
        assert_eq!(reply["type"], "turn_action", "reply = {reply}");
        assert_eq!(reply["request_id"], request_id, "reply = {reply}");
        assert_eq!(reply["action"], action, "reply = {reply}");
        match amount {
            Some(a) => assert_eq!(reply["params"]["amount"], a, "reply = {reply}"),
            None => {
                let empty = reply["params"]
                    .as_object()
                    .map(|m| m.is_empty())
                    .unwrap_or(true);
                assert!(empty, "params must be empty for {action}, got {reply}");
            }
        }
    }
}
