use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{crypto::hash::Digest, types::json_compatibility};
use casper_types::{
    auction::{
        Bid as AuctionBid, Bids as AuctionBids, EraId, ValidatorWeights as AuctionValidatorWeights,
    },
    U512,
};

/// Bids table.
pub type Bids = BTreeMap<json_compatibility::PublicKey, Bid>;
/// Validator weights by validator key.
pub type ValidatorWeights = BTreeMap<json_compatibility::PublicKey, U512>;

/// An entry in a founding validator map.
#[derive(PartialEq, Debug, Deserialize, Serialize, Clone)]
pub struct Bid {
    /// The purse that was used for bonding.
    pub bonding_purse: String,
    /// The total amount of staked tokens.
    pub staked_amount: U512,
    /// Delegation rate
    pub delegation_rate: u64,
    /// A flag that represents a winning entry.
    ///
    /// `Some` indicates locked funds for a specific era and an autowin status, and `None` case
    /// means that funds are unlocked and autowin status is removed.
    pub funds_locked: Option<u64>,
}

impl From<AuctionBid> for Bid {
    fn from(bid: AuctionBid) -> Self {
        Bid {
            bonding_purse: bid.bonding_purse.to_formatted_string(),
            staked_amount: bid.staked_amount,
            delegation_rate: bid.delegation_rate,
            funds_locked: bid.funds_locked,
        }
    }
}

/// Data structure summarizing auction contract data.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AuctionState {
    /// Global state hash
    pub global_state_hash: Digest,
    /// Era id.
    pub era_id: EraId,
    /// Validator weights for this era.
    pub validator_weights: Option<ValidatorWeights>,
    /// All bids.
    pub bids: Option<Bids>,
}

impl AuctionState {
    /// Create new instance of `AuctionState`
    pub fn new(
        global_state_hash: Digest,
        era_id: EraId,
        bids: Option<AuctionBids>,
        validator_weights: Option<AuctionValidatorWeights>,
    ) -> Self {
        let bids: Option<Bids> = {
            match bids {
                None => None,
                Some(items) => {
                    let mut ret: BTreeMap<json_compatibility::PublicKey, Bid> = BTreeMap::new();
                    for item in items {
                        let key = json_compatibility::PublicKey::from(item.0);
                        let value = item.1.into();
                        ret.insert(key, value);
                    }
                    Some(ret)
                }
            }
        };

        let validator_weights: Option<ValidatorWeights> = {
            match validator_weights {
                None => None,
                Some(items) => {
                    let mut ret: BTreeMap<json_compatibility::PublicKey, U512> = BTreeMap::new();
                    for item in items {
                        let key = json_compatibility::PublicKey::from(item.0);
                        let value = item.1;
                        ret.insert(key, value);
                    }
                    Some(ret)
                }
            }
        };

        AuctionState {
            global_state_hash,
            era_id,
            bids,
            validator_weights,
        }
    }
}
