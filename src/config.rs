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
        if self.artifacts.blueprint.trim().is_empty()
            || self.artifacts.rank_table.trim().is_empty()
            || self.artifacts.centroids.trim().is_empty()
        {
            return Err(BotError::Config(
                "blueprint, rank_table and centroids paths are required".into(),
            ));
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
                rank_table: "hand_ranks.bin".into(),
                centroids: "centroids.bin".into(),
                ..Default::default()
            },
            deterministic: false,
        }
    }

    #[test]
    fn accepts_valid_config() {
        assert!(cfg().validate().is_ok());
    }

    #[test]
    fn rejects_bad_url_token_and_paths() {
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
        assert!(c.validate().is_err());
    }
}
