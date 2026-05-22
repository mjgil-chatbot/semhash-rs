use crate::datamodels::{DeduplicationResult, DuplicateRecord};
use crate::error::{Result, SemHashError};
use crate::utils::{coerce_value, dict_record_to_text, to_key_map, DictRecord, Record, Value};
use std::collections::{BTreeMap, HashMap};

/// Exact duplicate output: `(duplicate_record, records_it_matches)`.
pub type ExactDuplicates = Vec<(DictRecord, Vec<DictRecord>)>;

/// A record paired with a perfect exact-duplicate score.
pub fn add_scores_to_records(records: &[DictRecord]) -> Vec<(DictRecord, f32)> {
    records
        .iter()
        .cloned()
        .map(|record| (record, 1.0))
        .collect()
}

/// Group records by exact match on columns, preserving first-occurrence order.
pub fn group_records_by_key(
    records: &[DictRecord],
    columns: &[String],
) -> Result<(Vec<DictRecord>, Vec<Vec<DictRecord>>)> {
    let mut buckets: HashMap<BTreeMap<String, Value>, usize> = HashMap::new();
    let mut items: Vec<Vec<DictRecord>> = Vec::new();

    for record in records {
        let key = to_key_map(record, columns)?;
        if let Some(index) = buckets.get(&key).copied() {
            items[index].push(record.clone());
        } else {
            buckets.insert(key, items.len());
            items.push(vec![record.clone()]);
        }
    }

    let deduplicated_records = items.iter().map(|bucket| bucket[0].clone()).collect();
    Ok((deduplicated_records, items))
}

/// Remove exact duplicates based on the hashable representation of each record.
///
/// When `reference_records` is `None`, duplicates are removed within `records`.
/// When a reference index is supplied, duplicates are removed only if they exist
/// in the reference items.
pub fn remove_exact_duplicates(
    records: &[DictRecord],
    columns: &[String],
    reference_records: Option<&[Vec<DictRecord>]>,
) -> Result<(Vec<DictRecord>, ExactDuplicates)> {
    let mut deduplicated = Vec::new();
    let mut duplicates = Vec::new();
    let mut seen: HashMap<BTreeMap<String, Value>, Vec<DictRecord>> = HashMap::new();

    if let Some(reference) = reference_records {
        for record_set in reference {
            if let Some(first) = record_set.first() {
                let key = to_key_map(first, columns)?;
                seen.insert(key, record_set.clone());
            }
        }
    }

    for record in records {
        let key = to_key_map(record, columns)?;
        if let Some(duplicated_records) = seen.get(&key) {
            duplicates.push((record.clone(), duplicated_records.clone()));
        } else {
            deduplicated.push(record.clone());
            if reference_records.is_none() {
                seen.entry(key).or_default().push(record.clone());
            }
        }
    }

    Ok((deduplicated, duplicates))
}

/// Validate and prepare public `Record` values for processing.
///
/// Text records are converted into dictionaries with a single `text` column.
/// Dictionary records require explicit columns and preserve non-embedding fields.
pub fn prepare_records(
    records: &[Record],
    columns: Option<Vec<String>>,
) -> Result<(Vec<DictRecord>, Vec<String>, bool)> {
    if records.is_empty() {
        return Err(SemHashError::EmptyRecords);
    }

    match &records[0] {
        Record::Text(_) => {
            if records
                .iter()
                .any(|record| !matches!(record, Record::Text(_)))
            {
                return Err(SemHashError::MixedRecordTypes {
                    expected: "strings",
                });
            }
            let columns = vec!["text".to_string()];
            let dict_records = records
                .iter()
                .map(|record| match record {
                    Record::Text(text) => {
                        let mut map = DictRecord::new();
                        map.insert("text".to_string(), Value::String(text.clone()));
                        map
                    }
                    Record::Dict(_) => unreachable!(),
                })
                .collect();
            Ok((dict_records, columns, true))
        }
        Record::Dict(_) => {
            let columns = columns.ok_or(SemHashError::ColumnsRequired)?;
            if records
                .iter()
                .any(|record| !matches!(record, Record::Dict(_)))
            {
                return Err(SemHashError::MixedRecordTypes { expected: "dicts" });
            }

            let mut dict_records = Vec::with_capacity(records.len());
            for record in records {
                let Record::Dict(map) = record else {
                    unreachable!()
                };
                let mut coerced = map.clone();
                for column in &columns {
                    let value = map.get(column).ok_or_else(|| SemHashError::MissingColumn {
                        column: column.clone(),
                    })?;
                    if matches!(value, Value::Null) {
                        return Err(SemHashError::NoneValue {
                            column: column.clone(),
                            record: Value::Map(map.clone()).canonical(),
                        });
                    }
                    coerced.insert(column.clone(), coerce_value(value));
                }
                dict_records.push(coerced);
            }
            Ok((dict_records, columns, false))
        }
    }
}

/// Convert a dict record into the same tab-separated text that Python uses for
/// string-mode result mapping.
pub fn dict_to_string(record: &DictRecord, columns: &[String]) -> String {
    dict_record_to_text(record, columns)
}

/// Convert a deduplication result from dictionary internals back into text
/// records when the SemHash instance was built from text records.
pub fn map_deduplication_result_to_strings(
    result: DeduplicationResult,
    columns: &[String],
) -> DeduplicationResult {
    let selected = result
        .selected
        .iter()
        .map(|record| match record {
            Record::Dict(map) => Record::Text(dict_to_string(map, columns)),
            Record::Text(text) => Record::Text(text.clone()),
        })
        .collect();

    let filtered = result
        .filtered
        .iter()
        .map(|dup_record| {
            let record = match &dup_record.record {
                Record::Dict(map) => Record::Text(dict_to_string(map, columns)),
                Record::Text(text) => Record::Text(text.clone()),
            };
            let duplicates = dup_record
                .duplicates
                .iter()
                .map(|(record, score)| {
                    let mapped = match record {
                        Record::Dict(map) => Record::Text(dict_to_string(map, columns)),
                        Record::Text(text) => Record::Text(text.clone()),
                    };
                    (mapped, *score)
                })
                .collect();
            DuplicateRecord {
                record,
                exact: dup_record.exact,
                duplicates,
            }
        })
        .collect();

    DeduplicationResult::new(selected, filtered, result.threshold, result.columns.clone())
}

pub(crate) fn validate_if_strings(
    records: &[Record],
    columns: &[String],
    was_string: bool,
) -> Result<Vec<DictRecord>> {
    if records.is_empty() {
        return Err(SemHashError::EmptyRecords);
    }

    match &records[0] {
        Record::Text(_) => {
            if !was_string {
                return Err(SemHashError::OriginallyNotStrings);
            }
            if records
                .iter()
                .any(|record| !matches!(record, Record::Text(_)))
            {
                return Err(SemHashError::MixedRecordTypes {
                    expected: "all strings",
                });
            }
            let mut out = Vec::with_capacity(records.len());
            for record in records {
                let Record::Text(text) = record else {
                    unreachable!()
                };
                let mut map = DictRecord::new();
                map.insert("text".to_string(), Value::String(text.clone()));
                out.push(map);
            }
            Ok(out)
        }
        Record::Dict(_) => {
            if records
                .iter()
                .any(|record| !matches!(record, Record::Dict(_)))
            {
                return Err(SemHashError::MixedRecordTypes {
                    expected: "all dictionaries",
                });
            }

            let mut result = Vec::with_capacity(records.len());
            for record in records {
                let Record::Dict(map) = record else {
                    unreachable!()
                };
                let mut out = map.clone();
                for column in columns {
                    let value = map.get(column).ok_or_else(|| SemHashError::MissingColumn {
                        column: column.clone(),
                    })?;
                    if matches!(value, Value::Null) {
                        return Err(SemHashError::NoneValue {
                            column: column.clone(),
                            record: Value::Map(map.clone()).canonical(),
                        });
                    }
                    out.insert(column.clone(), coerce_value(value));
                }
                result.push(out);
            }
            Ok(result)
        }
    }
}

pub(crate) fn dicts_to_records(records: Vec<DictRecord>) -> Vec<Record> {
    records.into_iter().map(Record::Dict).collect()
}

pub(crate) fn scored_dicts_to_records(records: Vec<(DictRecord, f32)>) -> Vec<(Record, f32)> {
    records
        .into_iter()
        .map(|(record, score)| (Record::Dict(record), score))
        .collect()
}

pub(crate) fn records_to_dicts_for_featurize(
    records: &[Record],
    columns: &[String],
) -> Result<Vec<DictRecord>> {
    records
        .iter()
        .map(|record| match record {
            Record::Dict(map) => Ok(map.clone()),
            Record::Text(text) => {
                let mut map = DictRecord::new();
                if columns.len() == 1 {
                    map.insert(columns[0].clone(), Value::String(text.clone()));
                    Ok(map)
                } else {
                    Err(SemHashError::Message(
                        "Text candidates can only be featurized with a single column".to_string(),
                    ))
                }
            }
        })
        .collect()
}
