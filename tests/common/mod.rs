//! Shared test helpers: build *synthetic, deterministic* pkr-sota artifacts.
//!
//! Determinism trick: `hand_ranks.bin` is filled with zeros, so every 5-card
//! evaluation returns rank 0 → hero always ties villain → every EHS sample is
//! exactly 0.5 (EHS² = 0.25) regardless of sampling. Cluster assignment is
//! therefore fully deterministic on every street, which makes infoset hashes
//! stable across runs and lets the integration tests assert exact actions.
//!
//! The blueprint is hand-assembled following the documented file layout
//! (`pkr-export/src/writer.rs::write_blueprint` +
//! `pkr-runtime/src/mmap.rs::MmapReader::new`).

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;

use pkr_abstraction::{save_centroids, CentroidStore, KMeansAbstraction};
use pkr_contracts::{AbstractionBuilder, Evaluator};
use pkr_eval::TableEvaluator;
use pkr_export::fmph::build_fmph;
use pkr_export::header::{
    FileHeader, FmphHeader, TranslationTableHeader, FORMAT_VERSION_V2,
    HASH_ALGO_FNV1A64_INFOSET,
};

use pkr_chipzen_bot::adapter::TurnContext;
use pkr_chipzen_bot::strategy::ArtifactPaths;

/// C(52,5)
const COMBOS_5: usize = 2_598_960;
/// C(52,3)
const COMBOS_3: usize = 22_100;

/// Compute the trainer-exact infoset hash for `ctx` using the synthetic
/// artifacts (must match what `Engine::load` will construct at runtime).
pub fn synthetic_hash(dir: &Path, ctx: &TurnContext) -> u64 {
    let evaluator: std::sync::Arc<TableEvaluator> = std::sync::Arc::new(
        TableEvaluator::new(dir.join("hand_ranks.bin")).expect("rank table"),
    );
    let store = pkr_abstraction::load_centroids(
        dir.join("centroids.bin").to_str().unwrap(),
    )
    .expect("centroids");
    let mut abstraction = KMeansAbstraction::from_store(store, evaluator.clone());
    abstraction
        .load_flop_buckets(dir.join("flop_buckets.bin").to_str().unwrap())
        .expect("flop buckets");
    let _ = &mut abstraction; // silence clippy when loads are optional
    let board = ctx.padded_board();
    let history = ctx.padded_history();
    abstraction.get_infoset_hash(&ctx.hole, &board, &history, ctx.street)
}

/// Write the base artifacts (rank table, centroids, flop buckets) without a
/// blueprint — needed before computing synthetic hashes.
pub fn build_base_artifacts(dir: &Path) {
    std::fs::create_dir_all(dir).expect("artifact dir");
    write_rank_table(dir);
    write_centroids(dir);
    write_flop_buckets(dir);
}

/// Write all artifacts into `dir` (base + blueprint for `entries`).
pub fn build_artifacts(dir: &Path, entries: &[(u64, [u8; 6])]) -> ArtifactPaths {
    build_base_artifacts(dir);
    write_blueprint(dir, entries);

    ArtifactPaths {
        blueprint: dir.join("blueprint.bin").display().to_string(),
        rank_table: dir.join("hand_ranks.bin").display().to_string(),
        centroids: dir.join("centroids.bin").display().to_string(),
        flop_buckets: Some(dir.join("flop_buckets.bin").display().to_string()),
        ..Default::default()
    }
}

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
    save_centroids(dir.join("centroids.bin").to_str().unwrap(), &store)
        .expect("save centroids");
}

/// Deterministic flop buckets: bucket = combo_index % 7.
fn write_flop_buckets(dir: &Path) {
    let path = dir.join("flop_buckets.bin");
    let file = std::fs::File::create(&path).expect("create flop_buckets.bin");
    let mut w = std::io::BufWriter::new(file);
    for i in 0..COMBOS_3 {
        w.write_all(&[(i % 7) as u8]).expect("write bucket");
    }
    w.flush().expect("flush flop buckets");
}

/// Hand-assemble a PKRSOTA1-v2 blueprint file for the given key→CDF entries.
pub fn write_blueprint(dir: &Path, entries: &[(u64, [u8; 6])]) {
    assert!(!entries.is_empty(), "FMph needs >= 1 key");
    let mut keys: Vec<u64> = entries.iter().map(|(k, _)| *k).collect();
    keys.sort_unstable();
    keys.dedup();

    let fmph = build_fmph(&keys);
    let fmph_header: FmphHeader = fmph.to_header();

    let file_header = FileHeader {
        magic: *b"PKRSOTA1",
        version: FORMAT_VERSION_V2,
        variant_id: 0,
        infoset_count: keys.len() as u64,
        max_actions_k: 6,
        hash_algo: HASH_ALGO_FNV1A64_INFOSET,
        _padding: [0u8; 6],
    };

    let ttable_header = TranslationTableHeader {
        num_entries: 0, // runtime lookup binary-searches keys; ttable unused
        action_size: 2,
        _padding: [0u8; 4],
    };

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(bytemuck::bytes_of(&file_header));
    buf.extend_from_slice(bytemuck::bytes_of(&fmph_header));
    for &d in &fmph.displacements {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    let cdf_size = (keys.len() * 6) as u32;
    buf.extend_from_slice(&cdf_size.to_le_bytes());
    buf.extend_from_slice(bytemuck::bytes_of(&ttable_header));

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

/// CDF helpers.
pub fn cdf_all_on(slot: usize) -> [u8; 6] {
    let mut cdf = [255u8; 6];
    for b in cdf.iter_mut().take(slot) {
        *b = 0;
    }
    cdf
}

/// Mock-evaluator parity check: with the all-zero table every hand ties.
pub struct ZeroEvaluator;
impl Evaluator for ZeroEvaluator {
    fn evaluate_hand(&self, _hole: &[u8], _board: &[u8]) -> u32 {
        0
    }
}
