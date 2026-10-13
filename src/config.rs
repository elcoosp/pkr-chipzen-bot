//! Runtime configuration for the bot.

use crate::error::{BotError, Result};
use crate::strategy::ArtifactPaths;

/// Everything the runner needs. Built from CLI flags + environment in
/// `main.rs` (CLI wins, mirroring the Python reference client).
#[derive(Debug, Clone)]
pub struct BotConfig {
    /// Platform origin, e.g. `wss://staging.chipzen.ai`.
    pub base_url: String,
    /// The external-API bot's UUID.
    pub bot_id: String,
    /// `cz_extbot_…` long-lived token.
    pub token: String,
    /// Keep cycling: reconnect the lobby after each match and play forever.
    pub loop_matches: bool,
    /// Trained artifact paths.
    pub artifacts: ArtifactPaths,
    /// Argmax instead of CDF sampling.
    pub deterministic: bool,
    /// Infoset signature generation (1–5). The shipped k7-baseline
    /// checkpoint was trained with `PKR_SIG_V5=1` → **5**.
    pub sig: u8,
    /// Preflop centroid count for the fingerprint check (k7-baseline: 200).
    pub preflop_k: u32,
    /// Use the EHS heuristic on blueprint misses instead of the engine's
    /// `fallback_advice()` mild policy.
    pub fallback_ehs: bool,
    /// Hard-fail when the flop/turn/river abstraction tables are missing
    /// (true for the CLI; the integration test relaxes it for synthetic
    /// artifacts without the 305 MB turn table).
    pub require_tables: bool,
}

impl BotConfig {
    pub fn validate(&self) -> Result<()> {
        let base = self.base_url.trim();
        if !(base.starts_with("wss://") || base.starts_with("ws://")) {
            return Err(BotError::Config(format!(
                "base_url must start with wss:// (or ws:// for localhost), got {base:?}"
            )));
        }
        if self.bot_id.trim().is_empty() {
            return Err(BotError::Config("bot_id must not be empty".into()));
        }
        if !self.token.starts_with("cz_extbot_") {
            return Err(BotError::Config(
                "token must be a cz_extbot_… token (see onboarding §2)".into(),
            ));
        }
        if !(1..=5).contains(&self.sig) {
            return Err(BotError::Config(format!(
                "--sig must be 1..=5, got {}",
                self.sig
            )));
        }
        if self.preflop_k == 0 {
            return Err(BotError::Config("--preflop-k must be positive".into()));
        }
        if self.artifacts.blueprint.trim().is_empty() && self.artifacts.ckpt.is_none() {
            return Err(BotError::Config(
                "either --blueprint or --ckpt must be provided".into(),
            ));
        }
        if self.artifacts.centroids.trim().is_empty() {
            return Err(BotError::Config("centroids path is required".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BotConfig {
        BotConfig {
            base_url: "wss://staging.chipzen.ai".into(),
            bot_id: "b0b".into(),
            token: "cz_extbot_abc".into(),
            loop_matches: false,
            artifacts: ArtifactPaths {
                blueprint: "blueprint.bin".into(),
                centroids: "centroids.bin".into(),
                ..Default::default()
            },
            deterministic: false,
            sig: 5,
            preflop_k: 200,
            fallback_ehs: false,
            require_tables: true,
        }
    }

    #[test]
    fn accepts_valid_config() {
        assert!(cfg().validate().is_ok());
        // Checkpoint-only mode is valid (blueprint omitted).
        let mut c = cfg();
        c.artifacts.blueprint = String::new();
        c.artifacts.ckpt = Some("train.ckpt".into());
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_bad_url_token_paths_and_sig() {
        let mut c = cfg();
        c.base_url = "https://staging.chipzen.ai".into();
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.token = "sk_live_nope".into();
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.bot_id = "  ".into();
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.artifacts.blueprint = "".into();
        assert!(c.validate().is_err(), "no blueprint and no ckpt");

        let mut c = cfg();
        c.sig = 9;
        assert!(c.validate().is_err());

        let mut c = cfg();
        c.preflop_k = 0;
        assert!(c.validate().is_err());
    }
}
