use crate::error::{Result, SemHashError};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Dense embedding matrix represented as rows of f32 vectors.
pub type Embeddings = Vec<Vec<f32>>;

/// Rust representation of Python's `dict[str, Any]` records.
pub type DictRecord = BTreeMap<String, Value>;

/// A supported record value.
///
/// Python SemHash accepts arbitrary `Any` values in dictionary records. Rust is
/// statically typed, so this enum covers the common primitives plus nested
/// lists/maps and byte payloads for non-text encoders.
#[derive(Clone, Debug)]
pub enum Value {
    String(String),
    Bytes(Vec<u8>),
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Null,
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    /// Stable, JSON-like representation used for hashing, display, and
    /// duplicate de-duplication inside result inspection.
    pub fn canonical(&self) -> String {
        match self {
            Self::String(s) => format!("\"{}\"", escape_json(s)),
            Self::Bytes(bytes) => {
                let mut out = String::from("bytes:");
                for byte in bytes {
                    out.push_str(&format!("{byte:02x}"));
                }
                out
            }
            Self::I64(v) => v.to_string(),
            Self::U64(v) => v.to_string(),
            Self::F64(v) => {
                if v.is_nan() {
                    "NaN".to_string()
                } else if v.is_infinite() && v.is_sign_positive() {
                    "Infinity".to_string()
                } else if v.is_infinite() {
                    "-Infinity".to_string()
                } else {
                    // Keep a stable round-trippable representation.
                    format!("{:?}", v)
                }
            }
            Self::Bool(v) => v.to_string(),
            Self::Null => "null".to_string(),
            Self::List(values) => {
                let parts: Vec<String> = values.iter().map(Value::canonical).collect();
                format!("[{}]", parts.join(","))
            }
            Self::Map(values) => {
                let parts: Vec<String> = values
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json(k), v.canonical()))
                    .collect();
                format!("{{{}}}", parts.join(","))
            }
        }
    }

    /// Lossy string conversion used when joining records back into text.
    pub fn as_string_lossy(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Bytes(bytes) => String::from_utf8_lossy(bytes).to_string(),
            Self::I64(v) => v.to_string(),
            Self::U64(v) => v.to_string(),
            Self::F64(v) => v.to_string(),
            Self::Bool(v) => {
                if *v {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            Self::Null => String::new(),
            Self::List(_) | Self::Map(_) => self.canonical(),
        }
    }

    fn tag(&self) -> u8 {
        match self {
            Self::String(_) => 0,
            Self::Bytes(_) => 1,
            Self::I64(_) => 2,
            Self::U64(_) => 3,
            Self::F64(_) => 4,
            Self::Bool(_) => 5,
            Self::Null => 6,
            Self::List(_) => 7,
            Self::Map(_) => 8,
        }
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<Vec<u8>> for Value {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for Value {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::I64(value as i64)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::U64(value as u64)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::F64(value as f64)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Null, Self::Null) => true,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag().hash(state);
        match self {
            Self::String(v) => v.hash(state),
            Self::Bytes(v) => v.hash(state),
            Self::I64(v) => v.hash(state),
            Self::U64(v) => v.hash(state),
            Self::F64(v) => v.to_bits().hash(state),
            Self::Bool(v) => v.hash(state),
            Self::Null => {}
            Self::List(v) => v.hash(state),
            Self::Map(v) => v.hash(state),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        let tag_cmp = self.tag().cmp(&other.tag());
        if tag_cmp != Ordering::Equal {
            return tag_cmp;
        }
        match (self, other) {
            (Self::String(a), Self::String(b)) => a.cmp(b),
            (Self::Bytes(a), Self::Bytes(b)) => a.cmp(b),
            (Self::I64(a), Self::I64(b)) => a.cmp(b),
            (Self::U64(a), Self::U64(b)) => a.cmp(b),
            (Self::F64(a), Self::F64(b)) => a.to_bits().cmp(&b.to_bits()),
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Null, Self::Null) => Ordering::Equal,
            (Self::List(a), Self::List(b)) => a.cmp(b),
            (Self::Map(a), Self::Map(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Bytes(bytes) => write!(f, "{}", String::from_utf8_lossy(bytes)),
            Self::I64(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Null => write!(f, "None"),
            Self::List(_) | Self::Map(_) => write!(f, "{}", self.canonical()),
        }
    }
}

/// Public record type. Text records correspond to Python strings; dictionary
/// records correspond to Python `dict[str, Any]` records.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Record {
    Text(String),
    Dict(DictRecord),
}

impl Record {
    pub fn text<S: Into<String>>(text: S) -> Self {
        Self::Text(text.into())
    }

    pub fn dict(record: DictRecord) -> Self {
        Self::Dict(record)
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Dict(_) => None,
        }
    }

    pub fn as_dict(&self) -> Option<&DictRecord> {
        match self {
            Self::Dict(value) => Some(value),
            Self::Text(_) => None,
        }
    }

    pub fn canonical(&self) -> String {
        match self {
            Self::Text(s) => format!("text:{}", s),
            Self::Dict(map) => Value::Map(map.clone()).canonical(),
        }
    }
}

impl From<&str> for Record {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for Record {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<DictRecord> for Record {
    fn from(value: DictRecord) -> Self {
        Self::Dict(value)
    }
}

impl Display for Record {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "{s}"),
            Self::Dict(map) => write!(f, "{}", Value::Map(map.clone()).canonical()),
        }
    }
}

/// Encoder protocol equivalent to Python's `Encoder` Protocol.
///
/// The input is one column at a time. Text encoders typically expect all values
/// to be `Value::String`; multimodal encoders may use `Bytes` or nested values.
pub trait Encoder: Send + Sync {
    fn encode(&self, inputs: &[Value]) -> Result<Embeddings>;
}

/// Deterministic fallback encoder used when no model is supplied.
///
/// Python SemHash loads `minishlab/potion-base-8M` by default. This Rust port
/// keeps the API callable without network/model files by defaulting to a stable
/// hashing encoder. Pass your own `Encoder` implementation to match a specific
/// embedding model.
#[derive(Debug, Clone)]
pub struct HashingEncoder {
    dimension: usize,
}

impl HashingEncoder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension: dimension.max(1),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for HashingEncoder {
    fn default() -> Self {
        Self::new(128)
    }
}

impl Encoder for HashingEncoder {
    fn encode(&self, inputs: &[Value]) -> Result<Embeddings> {
        let mut embeddings = Vec::with_capacity(inputs.len());
        for value in inputs {
            let text = value.as_string_lossy().to_lowercase();
            let mut vector = vec![0.0_f32; self.dimension];
            let mut emitted = false;

            for token in text.split_whitespace() {
                emitted = true;
                add_hashed_feature(token.as_bytes(), &mut vector);
                let chars: Vec<char> = token.chars().collect();
                for window in 3..=5 {
                    if chars.len() >= window {
                        for gram in chars.windows(window) {
                            let gram: String = gram.iter().collect();
                            add_hashed_feature(gram.as_bytes(), &mut vector);
                        }
                    }
                }
            }

            if !emitted {
                add_hashed_feature(text.as_bytes(), &mut vector);
            }

            normalize(&mut vector);
            embeddings.push(vector);
        }
        Ok(embeddings)
    }
}

/// Convert a value to a hashable representation. In Rust, `Value` is already
/// hashable; nested non-primitive values are canonicalized to strings to mirror
/// Python's fallback behavior for unhashable objects.
pub fn make_hashable(value: &Value) -> Value {
    match value {
        Value::List(_) | Value::Map(_) => Value::String(value.canonical()),
        _ => value.clone(),
    }
}

/// Coerce a value for encoding: strings and bytes are preserved, primitive
/// numbers/bools are stringified, and complex values pass through unchanged.
pub fn coerce_value(value: &Value) -> Value {
    match value {
        Value::String(_) | Value::Bytes(_) => value.clone(),
        Value::I64(_) | Value::U64(_) | Value::F64(_) | Value::Bool(_) => {
            Value::String(value.as_string_lossy())
        }
        Value::Null => Value::Null,
        Value::List(_) | Value::Map(_) => value.clone(),
    }
}

/// Convert a record to a column-only, hashable key map.
pub fn to_key_map(record: &DictRecord, columns: &[String]) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for column in columns {
        let value = record
            .get(column)
            .ok_or_else(|| SemHashError::MissingColumn {
                column: column.clone(),
            })?;
        out.insert(column.clone(), make_hashable(value));
    }
    Ok(out)
}

/// Python-compatible alias for `to_key_map`.
pub fn to_frozendict(record: &DictRecord, columns: &[String]) -> Result<BTreeMap<String, Value>> {
    to_key_map(record, columns)
}

/// Compute the `candidate_limit="auto"` value used by representative sampling.
pub fn compute_candidate_limit(
    total: usize,
    selection_size: usize,
    fraction: f32,
    min_candidates: usize,
    max_candidates: usize,
) -> usize {
    let mut limit = (total as f32 * fraction) as usize;
    limit = limit.max(selection_size);
    limit = limit.max(min_candidates);
    limit.min(max_candidates).min(total)
}

/// Default candidate limit matching Python defaults: fraction=0.1,
/// min_candidates=100, max_candidates=1000.
pub fn compute_candidate_limit_default(total: usize, selection_size: usize) -> usize {
    compute_candidate_limit(total, selection_size, 0.1, 100, 1000)
}

/// Featurize records one column at a time and concatenate the resulting
/// embeddings horizontally, matching Python `featurize`.
pub fn featurize(
    records: &[DictRecord],
    columns: &[String],
    model: &dyn Encoder,
) -> Result<Embeddings> {
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let mut embeddings_per_col: Vec<Embeddings> = Vec::with_capacity(columns.len());
    for column in columns {
        let mut column_values = Vec::with_capacity(records.len());
        for record in records {
            let value = record
                .get(column)
                .ok_or_else(|| SemHashError::MissingColumn {
                    column: column.clone(),
                })?;
            column_values.push(value.clone());
        }

        let column_embeddings = model.encode(&column_values).map_err(|err| {
            let sample_type = column_values
                .first()
                .map(value_type_name)
                .unwrap_or_else(|| "unknown".to_string());
            SemHashError::EncodeError(format!(
                "Failed to encode column '{column}' (data type: {sample_type}). If encoding non-text data, provide a compatible encoder via the `model` parameter. See the SemHash documentation for more info. Source error: {err}"
            ))
        })?;
        validate_embeddings_shape(&column_embeddings, records.len(), None)?;
        embeddings_per_col.push(column_embeddings);
    }

    let mut out = vec![Vec::new(); records.len()];
    for col_embeddings in embeddings_per_col {
        for (row, values) in out.iter_mut().zip(col_embeddings) {
            row.extend(values);
        }
    }
    Ok(out)
}

pub(crate) fn validate_embeddings_shape(
    embeddings: &Embeddings,
    expected_rows: usize,
    expected_cols: Option<usize>,
) -> Result<usize> {
    if embeddings.len() != expected_rows {
        return Err(SemHashError::InvalidEmbeddings(format!(
            "Number of embeddings ({}) must match number of records ({expected_rows})",
            embeddings.len()
        )));
    }
    let Some(first) = embeddings.first() else {
        return Ok(0);
    };
    if first.is_empty() {
        return Err(SemHashError::InvalidEmbeddings(
            "embeddings must be a 2D array with non-empty rows".to_string(),
        ));
    }
    let cols = first.len();
    if let Some(expected) = expected_cols {
        if cols != expected {
            return Err(SemHashError::InvalidEmbeddings(format!(
                "embeddings must have {expected} columns, got {cols}"
            )));
        }
    }
    for row in embeddings {
        if row.len() != cols {
            return Err(SemHashError::InvalidEmbeddings(
                "embeddings must be a rectangular 2D array".to_string(),
            ));
        }
    }
    Ok(cols)
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

pub(crate) fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

pub(crate) fn dict_record_to_text(record: &DictRecord, columns: &[String]) -> String {
    columns
        .iter()
        .map(|column| {
            record
                .get(column)
                .map(Value::as_string_lossy)
                .unwrap_or_default()
                .replace('\t', " ")
        })
        .collect::<Vec<_>>()
        .join("\t")
}

pub(crate) fn record_key_for_selected(record: &Record, columns: Option<&[String]>) -> String {
    match (record, columns) {
        (Record::Dict(map), Some(cols)) => match to_key_map(map, cols) {
            Ok(key) => Value::Map(key).canonical(),
            Err(_) => record.canonical(),
        },
        _ => record.canonical(),
    }
}

fn add_hashed_feature(bytes: &[u8], vector: &mut [f32]) {
    let hash = fnv1a64(bytes);
    let idx = (hash as usize) % vector.len();
    let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
    vector[idx] += sign;
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn value_type_name(value: &Value) -> String {
    match value {
        Value::String(_) => "str".to_string(),
        Value::Bytes(_) => "bytes".to_string(),
        Value::I64(_) | Value::U64(_) => "int".to_string(),
        Value::F64(_) => "float".to_string(),
        Value::Bool(_) => "bool".to_string(),
        Value::Null => "None".to_string(),
        Value::List(_) => "list".to_string(),
        Value::Map(_) => "dict".to_string(),
    }
}

fn escape_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
