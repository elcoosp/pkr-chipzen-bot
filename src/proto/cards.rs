//! Card-string mapping between the chipzen wire format ("Ah", "Td", "2c", …)
//! and the pkr-sota engine byte encoding.
//!
//! Engine encoding (see `pkr-eval/src/slow.rs` header comment and
//! `pkr-core/src/card.rs` discriminants):
//!
//! ```text
//! card_id = suit_index * 13 + rank_index
//! suit_index: 0 = Spade, 1 = Heart, 2 = Diamond, 3 = Club
//! rank_index: 0 = Two, 1 = Three, ..., 8 = Ten, 9 = Jack, 10 = Queen,
//!             11 = King, 12 = Ace
//! ```
//!
//! Chipzen wire encoding (Layer-2 poker protocol §1): rank char first, suit
//! char second, e.g. `Ah` = Ace of hearts, `2c` = Two of clubs.

use crate::error::{BotError, Result};

/// Chipzen suit character → engine suit index.
/// Chipzen: `c` clubs, `d` diamonds, `h` hearts, `s` spades.
fn suit_index(c: u8) -> Result<u8> {
    match c {
        b's' => Ok(0), // Spade
        b'h' => Ok(1), // Heart
        b'd' => Ok(2), // Diamond
        b'c' => Ok(3), // Club
        other => Err(BotError::Protocol(format!(
            "invalid suit char '{}' in card string",
            other as char
        ))),
    }
}

/// Chipzen rank character → engine rank index (Two = 0 … Ace = 12).
fn rank_index(c: u8) -> Result<u8> {
    match c {
        b'2'..=b'9' => Ok(c - b'2'),
        b'T' => Ok(8),
        b'J' => Ok(9),
        b'Q' => Ok(10),
        b'K' => Ok(11),
        b'A' => Ok(12),
        other => Err(BotError::Protocol(format!(
            "invalid rank char '{}' in card string",
            other as char
        ))),
    }
}

/// Parse a chipzen card string (e.g. `"Ah"`) into the engine byte encoding.
pub fn parse_card(s: &str) -> Result<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(BotError::Protocol(format!(
            "card string must be 2 chars (rank+suit), got {s:?}"
        )));
    }
    // Wire order: rank first, then suit.
    let rank = rank_index(bytes[0])?;
    let suit = suit_index(bytes[1])?;
    Ok(suit * 13 + rank)
}

/// Parse a list of card strings.
pub fn parse_cards(cards: &[String]) -> Result<Vec<u8>> {
    cards.iter().map(|s| parse_card(s)).collect()
}

/// Betting phase → engine street code (0 = preflop … 3 = river).
pub fn phase_to_street(phase: &str) -> Result<u8> {
    match phase {
        "preflop" => Ok(0),
        "flop" => Ok(1),
        "turn" => Ok(2),
        "river" => Ok(3),
        other => Err(BotError::Protocol(format!("unknown phase {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_across_all_suits_and_ranks() {
        // Ace of spades = suit 0 * 13 + rank 12
        assert_eq!(parse_card("As").unwrap(), 12);
        // Ace of hearts = suit 1 * 13 + 12 = 25
        assert_eq!(parse_card("Ah").unwrap(), 25);
        // Ace of diamonds = suit 2 * 13 + 12 = 38
        assert_eq!(parse_card("Ad").unwrap(), 38);
        // Ace of clubs = suit 3 * 13 + 12 = 51
        assert_eq!(parse_card("Ac").unwrap(), 51);
        // Deuce of spades = 0
        assert_eq!(parse_card("2s").unwrap(), 0);
        // Deuce of clubs = 39
        assert_eq!(parse_card("2c").unwrap(), 39);
        // Ten of diamonds = 2 * 13 + 8 = 34
        assert_eq!(parse_card("Td").unwrap(), 34);
        // King of hearts = 13 + 11 = 24
        assert_eq!(parse_card("Kh").unwrap(), 24);
    }

    #[test]
    fn rejects_malformed_cards() {
        assert!(parse_card("hA").is_err(), "suit/rank swapped is invalid");
        assert!(parse_card("Ax").is_err(), "bad suit");
        assert!(parse_card("1h").is_err(), "bad rank");
        assert!(parse_card("A").is_err(), "too short");
        assert!(parse_card("Ahh").is_err(), "too long");
    }

    #[test]
    fn parses_card_lists() {
        let cards = vec!["Ah".to_string(), "Kd".to_string(), "2c".to_string()];
        assert_eq!(parse_cards(&cards).unwrap(), vec![25, 2 * 13 + 11, 39]);
    }

    #[test]
    fn maps_phases_to_streets() {
        assert_eq!(phase_to_street("preflop").unwrap(), 0);
        assert_eq!(phase_to_street("flop").unwrap(), 1);
        assert_eq!(phase_to_street("turn").unwrap(), 2);
        assert_eq!(phase_to_street("river").unwrap(), 3);
        assert!(phase_to_street("showdown").is_err());
    }
}
