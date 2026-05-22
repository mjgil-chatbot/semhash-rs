use crate::error::{Result, SemHashError};
use crate::utils::{record_key_for_selected, Record};
use std::collections::HashMap;

/// A duplicate record and the records that caused it to be considered a
/// duplicate.
#[derive(Clone, Debug, PartialEq)]
pub struct DuplicateRecord {
    pub record: Record,
    pub exact: bool,
    pub duplicates: Vec<(Record, f32)>,
}

impl DuplicateRecord {
    pub fn new(record: Record, exact: bool, duplicates: Vec<(Record, f32)>) -> Self {
        Self {
            record,
            exact,
            duplicates,
        }
    }

    /// Rethreshold the duplicates in-place, matching Python `_rethreshold`.
    pub fn rethreshold_in_place(&mut self, threshold: f32) {
        self.duplicates.retain(|(_, score)| *score >= threshold);
    }
}

/// A selected record with the filtered duplicates that map back to it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedWithDuplicates {
    pub record: Record,
    pub duplicates: Vec<(Record, f32)>,
}

impl SelectedWithDuplicates {
    pub fn new(record: Record, duplicates: Vec<(Record, f32)>) -> Self {
        Self { record, duplicates }
    }
}

/// Deduplication result.
#[derive(Clone, Debug, PartialEq)]
pub struct DeduplicationResult {
    pub selected: Vec<Record>,
    pub filtered: Vec<DuplicateRecord>,
    pub threshold: f32,
    pub columns: Option<Vec<String>>,
    selected_with_duplicates_cache: Option<Vec<SelectedWithDuplicates>>,
}

impl DeduplicationResult {
    pub fn new(
        selected: Vec<Record>,
        filtered: Vec<DuplicateRecord>,
        threshold: f32,
        columns: Option<Vec<String>>,
    ) -> Self {
        Self {
            selected,
            filtered,
            threshold,
            columns,
            selected_with_duplicates_cache: None,
        }
    }

    /// Return the percentage of records dropped.
    pub fn duplicate_ratio(&self) -> f32 {
        let denom = self.selected.len() + self.filtered.len();
        if denom == 0 {
            0.0
        } else {
            1.0 - self.selected.len() as f32 / denom as f32
        }
    }

    /// Return the percentage of records dropped because of an exact match.
    pub fn exact_duplicate_ratio(&self) -> f32 {
        let denom = self.selected.len() + self.filtered.len();
        if denom == 0 {
            0.0
        } else {
            self.filtered.iter().filter(|dup| dup.exact).count() as f32 / denom as f32
        }
    }

    /// Return the N least-similar duplicate pairs as
    /// `(original_record, duplicate_record, score)`.
    pub fn get_least_similar_from_duplicates(&self, n: usize) -> Vec<(Record, Record, f32)> {
        let mut all_pairs: Vec<(Record, Record, f32)> = self
            .filtered
            .iter()
            .flat_map(|dup| {
                dup.duplicates
                    .iter()
                    .map(move |(record, score)| (dup.record.clone(), record.clone(), *score))
            })
            .collect();
        all_pairs.sort_by(|a, b| a.2.total_cmp(&b.2));
        all_pairs.truncate(n);
        all_pairs
    }

    /// Rethreshold duplicates. Like Python SemHash, thresholds may only be
    /// raised; lowering a threshold requires re-running deduplication.
    pub fn rethreshold(&mut self, threshold: f32) -> Result<()> {
        if self.threshold > threshold {
            return Err(SemHashError::ThresholdTooSmall);
        }
        self.selected_with_duplicates_cache = None;

        let mut kept_filtered = Vec::new();
        for mut dup in self.filtered.drain(..) {
            dup.rethreshold_in_place(threshold);
            if dup.duplicates.is_empty() {
                self.selected.push(dup.record);
            } else {
                kept_filtered.push(dup);
            }
        }
        self.filtered = kept_filtered;
        self.threshold = threshold;
        Ok(())
    }

    /// For every kept record, return the duplicates that were removed along
    /// with their similarity scores. The result is cached and invalidated by
    /// `rethreshold`.
    pub fn selected_with_duplicates(&mut self) -> &[SelectedWithDuplicates] {
        if self.selected_with_duplicates_cache.is_none() {
            let columns = self.columns.as_deref();
            let mut buckets: HashMap<String, Vec<(Record, f32)>> = HashMap::new();

            for duplicate_record in &self.filtered {
                for (original_record, score) in &duplicate_record.duplicates {
                    let key = record_key_for_selected(original_record, columns);
                    buckets
                        .entry(key)
                        .or_default()
                        .push((duplicate_record.record.clone(), *score));
                }
            }

            let mut result = Vec::with_capacity(self.selected.len());
            for selected in &self.selected {
                let key = record_key_for_selected(selected, columns);
                let raw_list = buckets.get(&key).cloned().unwrap_or_default();
                let duplicates = dedupe_preserving_first_position(raw_list);
                result.push(SelectedWithDuplicates::new(selected.clone(), duplicates));
            }

            self.selected_with_duplicates_cache = Some(result);
        }

        self.selected_with_duplicates_cache
            .as_deref()
            .unwrap_or(&[])
    }

    /// Convenience conversion for string-mode deduplication results.
    pub fn selected_texts(&self) -> Option<Vec<String>> {
        self.selected
            .iter()
            .map(|record| record.as_text().map(ToOwned::to_owned))
            .collect()
    }
}

/// Result of filtering and representative sampling operations.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterResult {
    pub selected: Vec<Record>,
    pub filtered: Vec<Record>,
    pub scores_selected: Vec<f32>,
    pub scores_filtered: Vec<f32>,
}

impl FilterResult {
    pub fn new(
        selected: Vec<Record>,
        filtered: Vec<Record>,
        scores_selected: Vec<f32>,
        scores_filtered: Vec<f32>,
    ) -> Self {
        Self {
            selected,
            filtered,
            scores_selected,
            scores_filtered,
        }
    }

    /// Return the percentage of records filtered out.
    pub fn filter_ratio(&self) -> f32 {
        let denom = self.selected.len() + self.filtered.len();
        if denom == 0 {
            0.0
        } else {
            self.filtered.len() as f32 / denom as f32
        }
    }

    /// Return the percentage of records selected.
    pub fn selected_ratio(&self) -> f32 {
        1.0 - self.filter_ratio()
    }
}

fn dedupe_preserving_first_position(raw_list: Vec<(Record, f32)>) -> Vec<(Record, f32)> {
    let mut positions: HashMap<String, usize> = HashMap::new();
    let mut out: Vec<(Record, f32)> = Vec::new();

    for (record, score) in raw_list {
        let key = record.canonical();
        if let Some(position) = positions.get(&key).copied() {
            out[position] = (record, score);
        } else {
            positions.insert(key, out.len());
            out.push((record, score));
        }
    }
    out
}
