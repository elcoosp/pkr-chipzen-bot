//! Shared test helpers: build *synthetic, deterministic* pkr-sota artifacts
//! for the current engine generation (blueprint format v4, K = 7).
//!
//! Determinism trick: `hand_ranks.bin` is filled with zeros, so every 5-card
//! evaluation returns rank 0 → hero always ties villain → every EHS sample
//! is exactly 0.5 (EHS² = 0.25) regardless of sampling. Cluster assignment
//! is therefore fully deterministic on every street, which makes infoset
//! hashes stable across runs and lets the integration tests assert exact
//! actions.
//!
//! The blueprint is hand-assembled following the documented v4 file layout
//! (`pkr-export/src/writer.rs::write_blueprint`):
//! `[FileHeader:32][AnchorsSection:64][Fingerprint:40][counts][keys][cdf]`
//! with `max_actions_k = 7` and 7-byte quantized CDFs.
//!
//! Signature note: the fingerprint and every hash depend on the engine's
//! `PKR_SIG_*` env dispatch. All tests run under sig v5 (the shipped
//! k7-baseline configuration) — `pin_sig_env()` must run before anything
//! hashes.

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;

use pkr_abstraction::{save_centroids, CentroidStore};
use pkr_contracts::AbstractionBuilder;
use pkr_eval::TableEvaluator;
use pkr_export::header::{
    AnchorsSection, FileHeader, FORMAT_VERSION_V4, HASH_ALGO_FNV1A64_INFOSET,
};

use pkr_chipzen_bot::adapter::{infoset_signature, replay_hand, TurnContext, K};
use pkr_chipzen_bot::strategy::ArtifactPaths;

/// C(52,5)
const COMBOS_5: usize = 2_598_960;
/// C(52,5) * 10 — flop abstraction table (`init_table(1, …)` format).
const FLOP_IDX_SPACE: usize = 2_598_960 * 10;

/// Pin the engine's signature env to the shipped configuration (sig v5).
/// Idempotent; must run before any fingerprint or hash computation.
pub fn pin_sig_env() {
    pkr_chipzen_bot::apply_sig_env(5);
}

/// The trainer-exact infoset hash for `ctx` using the synthetic artifacts:
/// replay the wire hand into the engine `GameState`, then hash with the
/// same inputs `Engine::decide` uses (signature bytes + valid board slice).
pub fn synthetic_hash(dir: &Path, ctx: &TurnContext) -> u64 {
    pin_sig_env();
    let evaluator: std::sync::Arc<TableEvaluator> =
        std::sync::Arc::new(TableEvaluator::new(dir.join("hand_ranks.bin")).expect("rank table"));
    let store = pkr_abstraction::load_centroids(dir.join("centroids.bin").to_str().unwrap())
        .expect("centroids");
    let abstraction = pkr_abstraction::KMeansAbstraction::from_store(store, evaluator.clone());
    abstraction
        .init_table(1, dir.join("flop_table.bin").to_str().unwrap())
        .expect("flop table");
    abstraction
        .init_table(3, dir.join("river_table.bin").to_str().unwrap())
        .expect("river table");

    let rep = replay_hand(&ctx.state, &ctx.config).expect("replay");
    let st = &rep.state;
    let hole = st.hole[rep.our_seat as usize];
    let sig = infoset_signature(st);
    abstraction.get_infoset_hash(
        &hole,
        &st.board[..st.board_len as usize],
        &sig,
        st.street as u8,
    )
}

/// Write the base artifacts (rank table, centroids, street tables) without
/// a blueprint — needed before computing synthetic hashes. Also pins the
/// sig env.
pub fn build_base_artifacts(dir: &Path) {
    pin_sig_env();
    std::fs::create_dir_all(dir).expect("artifact dir");
    write_rank_table(dir);
    write_centroids(dir);
    write_flop_table(dir);
    write_river_table(dir);
}

/// Write all artifacts into `dir` (base + blueprint for `entries`).
pub fn build_artifacts(dir: &Path, entries: &[(u64, [u8; K])]) -> ArtifactPaths {
    build_base_artifacts(dir);
    write_blueprint(dir, entries);

    ArtifactPaths {
        blueprint: dir.join("blueprint.bin").display().to_string(),
        rank_table: Some(dir.join("hand_ranks.bin").display().to_string()),
        centroids: dir.join("centroids.bin").display().to_string(),
        flop_table: Some(dir.join("flop_table.bin").display().to_string()),
        river_table: Some(dir.join("river_table.bin").display().to_string()),
        // No synthetic 305 MB turn table: the turn hash arm falls back to
        // the (deterministic, zero-table) Monte-Carlo EHS cluster. Tests
        // must pass `require_tables = false`.
        ..Default::default()
    }
}

/// The number of preflop centroids in the synthetic store (feeds
/// `EngineOptions::preflop_k` and the fingerprint).
pub const SYNTHETIC_PREFLOP_K: u32 = 3;

/// All-zero rank table: every 5-card hand ranks 0 (ties everywhere → EHS 0.5).
fn write_rank_table(dir: &Path) {
    let path = dir.join("hand_ranks.bin");
    let file = std::fs::File::create(&path).expect("create hand_ranks.bin");
    let mut w = std::io::BufWriter::new(file);
    let zeros = [0u8; 4];
    for _ in 0..COMBOS_5 {
        w.write_all(&zeros).expect("write rank");
    }
    w.flush().expect("flush rank table");
}

fn write_centroids(dir: &Path) {
    let store = CentroidStore {
        centroids: vec![(0.2, 0.04), (0.5, 0.25), (0.8, 0.64)],
    };
    save_centroids(dir.join("centroids.bin").to_str().unwrap(), &store).expect("save centroids");
}

/// Deterministic flop abstraction table: bucket = (idx / 10) % 7.
fn write_flop_table(dir: &Path) {
    let path = dir.join("flop_table.bin");
    let file = std::fs::File::create(&path).expect("create flop_table.bin");
    let mut w = std::io::BufWriter::new(file);
    // Write in chunks to keep the writer fast.
    let mut chunk = Vec::with_capacity(1 << 16);
    for i in 0..FLOP_IDX_SPACE {
        chunk.push(((i / 10) % 7) as u8);
        if chunk.len() == chunk.capacity() {
            w.write_all(&chunk).expect("write flop chunk");
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        w.write_all(&chunk).expect("write flop tail");
    }
    w.flush().expect("flush flop table");
}

/// Deterministic river board-bucket table: bucket = idx % 7 over C(52,5).
fn write_river_table(dir: &Path) {
    let path = dir.join("river_table.bin");
    let file = std::fs::File::create(&path).expect("create river_table.bin");
    let mut w = std::io::BufWriter::new(file);
    let mut chunk = Vec::with_capacity(1 << 16);
    for i in 0..COMBOS_5 {
        chunk.push((i % 7) as u8);
        if chunk.len() == chunk.capacity() {
            w.write_all(&chunk).expect("write river chunk");
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        w.write_all(&chunk).expect("write river tail");
    }
    w.flush().expect("flush river table");
}

/// Hand-assemble a PKRSOTA1-v4 blueprint file for the given key→CDF entries.
///
/// v4 layout:
/// `[FileHeader:32][AnchorsSection:64][Fingerprint:40][key_count:u32][cdf_size:u32][keys][cdf]`
/// Matches `pkr-export/src/writer.rs::write_blueprint` exactly (K = 7,
/// fingerprint from the current engine constants under the pinned sig env).
pub fn write_blueprint(dir: &Path, entries: &[(u64, [u8; K])]) {
    pin_sig_env();
    assert!(!entries.is_empty(), "need at least 1 key");
    let mut keys: Vec<u64> = entries.iter().map(|(k, _)| *k).collect();
    keys.sort_unstable();
    keys.dedup();

    let file_header = FileHeader {
        magic: *b"PKRSOTA1",
        version: FORMAT_VERSION_V4,
        variant_id: 0,
        infoset_count: keys.len() as u64,
        max_actions_k: K as u8,
        hash_algo: HASH_ALGO_FNV1A64_INFOSET,
        _padding: [0u8; 6],
    };

    let anchors = AnchorsSection {
        anchors: pkr_export::header::ANCHORS,
    };
    let fingerprint = pkr_core::abstraction::AbstractionFingerprint::from_constants(
        SYNTHETIC_PREFLOP_K,
    );

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(bytemuck::bytes_of(&file_header));
    buf.extend_from_slice(bytemuck::bytes_of(&anchors));
    buf.extend_from_slice(bytemuck::bytes_of(&fingerprint));
    buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    let cdf_size = (keys.len() * K) as u32;
    buf.extend_from_slice(&cdf_size.to_le_bytes());

    // Sorted keys…
    for &k in &keys {
        buf.extend_from_slice(&k.to_le_bytes());
    }
    // …then their CDFs (index-aligned with the sorted key array).
    for &k in &keys {
        let &(_, cdf) = entries.iter().find(|(key, _)| *key == k).expect("entry");
        for b in cdf {
            buf.push(b);
        }
    }

    std::fs::write(dir.join("blueprint.bin"), &buf).expect("write blueprint");
}

/// CDF helpers: all mass on one abstract slot (K = 7).
pub fn cdf_all_on(slot: usize) -> [u8; K] {
    let mut cdf = [255u8; K];
    for b in cdf.iter_mut().take(slot) {
        *b = 0;
    }
    cdf
}
