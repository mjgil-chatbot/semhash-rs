//! Strict Rust port of the public SemHash API.
//!
//! The crate mirrors the Python package layout with modules for `datamodels`,
//! `index`, `records`, `semhash`, `strategy`, and `utils`.

pub mod datamodels;
pub mod error;
pub mod index;
pub mod records;
pub mod semhash;
pub mod strategy;
pub mod utils;

pub use datamodels::{DeduplicationResult, DuplicateRecord, FilterResult, SelectedWithDuplicates};
pub use error::{Result, SemHashError};
pub use index::{Backend, DocScore, DocScores, Index, SingleQueryResult};
pub use semhash::{CandidateLimit, SemHash, SemHashOptions};
pub use strategy::{diversify, DiversifyResult, Strategy};
pub use utils::{
    coerce_value, compute_candidate_limit, compute_candidate_limit_default, featurize,
    make_hashable, to_frozendict, to_key_map, DictRecord, Embeddings, Encoder, HashingEncoder,
    Record, Value,
};
