use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SemHashError>;

/// Errors returned by the Rust SemHash port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemHashError {
    /// The caller passed no records.
    EmptyRecords,
    /// Columns were required but not supplied.
    ColumnsRequired,
    /// A record collection mixed text and dictionary records.
    MixedRecordTypes { expected: &'static str },
    /// A text collection was passed to an instance built from dictionaries.
    OriginallyNotStrings,
    /// A record is missing a requested column.
    MissingColumn { column: String },
    /// A record contains a null value in a column that must be encoded.
    NoneValue { column: String, record: String },
    /// Embeddings are malformed or do not match the records.
    InvalidEmbeddings(String),
    /// Encoder returned an invalid matrix.
    EncodeError(String),
    /// Threshold values cannot be lowered after deduplication.
    ThresholdTooSmall,
    /// Percentages must be between 0 and 1 inclusive.
    InvalidPercentage { name: &'static str },
    /// Any other user-facing error.
    Message(String),
}

impl Display for SemHashError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRecords => write!(f, "records must not be empty"),
            Self::ColumnsRequired => {
                write!(f, "Columns must be specified when passing dictionaries.")
            }
            Self::MixedRecordTypes { expected } => write!(f, "All records must be {expected}."),
            Self::OriginallyNotStrings => {
                write!(
                    f,
                    "Records were not originally strings, but you passed strings."
                )
            }
            Self::MissingColumn { column } => write!(f, "Missing column '{column}'"),
            Self::NoneValue { column, record } => {
                write!(f, "Column '{column}' has None value in record {record}")
            }
            Self::InvalidEmbeddings(msg) => write!(f, "{msg}"),
            Self::EncodeError(msg) => write!(f, "{msg}"),
            Self::ThresholdTooSmall => write!(f, "Threshold is smaller than the given value."),
            Self::InvalidPercentage { name } => write!(f, "{name} must be between 0 and 1"),
            Self::Message(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for SemHashError {}
