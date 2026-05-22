use crate::datamodels::{DeduplicationResult, DuplicateRecord, FilterResult};
use crate::error::{Result, SemHashError};
use crate::index::{Backend, Index};
use crate::records::{
    add_scores_to_records, dicts_to_records, group_records_by_key,
    map_deduplication_result_to_strings, prepare_records, records_to_dicts_for_featurize,
    remove_exact_duplicates, scored_dicts_to_records, validate_if_strings,
};
use crate::strategy::{diversify, Strategy};
use crate::utils::{
    compute_candidate_limit_default, featurize, to_key_map, validate_embeddings_shape, DictRecord,
    Embeddings, Encoder, HashingEncoder, Record, Value,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Candidate limit used by representative sampling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CandidateLimit {
    #[default]
    Auto,
    Value(usize),
}

impl From<usize> for CandidateLimit {
    fn from(value: usize) -> Self {
        Self::Value(value)
    }
}

/// Constructor options that correspond to Python keyword arguments.
#[derive(Clone, Default)]
pub struct SemHashOptions {
    pub columns: Option<Vec<String>>,
    pub model: Option<Arc<dyn Encoder>>,
    pub ann_backend: Backend,
}

impl SemHashOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.columns = Some(columns.into_iter().map(Into::into).collect());
        self
    }

    pub fn model(mut self, model: Arc<dyn Encoder>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn ann_backend(mut self, backend: Backend) -> Self {
        self.ann_backend = backend;
        self
    }
}

/// Main SemHash type.
pub struct SemHash {
    pub index: Index,
    pub model: Arc<dyn Encoder>,
    pub columns: Vec<String>,
    was_string: bool,
    ranking_cache: Mutex<Option<FilterResult>>,
}

impl SemHash {
    /// Initialize a SemHash instance from records.
    ///
    /// This mirrors Python `SemHash.from_records(records, columns=None,
    /// model=None, ann_backend=Backend.USEARCH, **kwargs)`. Rust does not have
    /// keyword/default arguments, so pass `SemHashOptions::default()` for the
    /// Python defaults.
    pub fn from_records(records: Vec<Record>, options: SemHashOptions) -> Result<Self> {
        let (dict_records, columns, was_string) = prepare_records(&records, options.columns)?;
        let model: Arc<dyn Encoder> = options
            .model
            .unwrap_or_else(|| Arc::new(HashingEncoder::default()));

        let (deduplicated_records, items) = group_records_by_key(&dict_records, &columns)?;
        let embeddings = featurize(&deduplicated_records, &columns, model.as_ref())?;
        let index = Index::from_vectors_and_items(embeddings, items, options.ann_backend)?;

        Ok(Self {
            index,
            model,
            columns,
            was_string,
            ranking_cache: Mutex::new(None),
        })
    }

    /// Convenience constructor for text records with default options.
    pub fn from_texts<I, S>(records: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let records = records
            .into_iter()
            .map(|s| Record::Text(s.into()))
            .collect();
        Self::from_records(records, SemHashOptions::default())
    }

    /// Initialize a SemHash instance from pre-computed embeddings.
    pub fn from_embeddings(
        embeddings: Embeddings,
        records: Vec<Record>,
        model: Arc<dyn Encoder>,
        options: SemHashOptions,
    ) -> Result<Self> {
        if records.is_empty() {
            return Err(SemHashError::EmptyRecords);
        }
        if embeddings.is_empty() || embeddings.first().is_none_or(Vec::is_empty) {
            return Err(SemHashError::InvalidEmbeddings(
                "embeddings must be a 2D array, got an empty or 1D shape".to_string(),
            ));
        }
        validate_embeddings_shape(&embeddings, records.len(), None)?;

        let (dict_records, columns, was_string) = prepare_records(&records, options.columns)?;

        let mut items: Vec<Vec<DictRecord>> = Vec::new();
        let mut keep_embedding_indices: Vec<usize> = Vec::new();
        let mut key_to_item_idx: HashMap<std::collections::BTreeMap<String, Value>, usize> =
            HashMap::new();

        for (i, record) in dict_records.iter().enumerate() {
            let key = to_key_map(record, &columns)?;
            if let Some(item_idx) = key_to_item_idx.get(&key).copied() {
                items[item_idx].push(record.clone());
            } else {
                key_to_item_idx.insert(key, items.len());
                items.push(vec![record.clone()]);
                keep_embedding_indices.push(i);
            }
        }

        let deduplicated_embeddings = keep_embedding_indices
            .into_iter()
            .map(|idx| embeddings[idx].clone())
            .collect();
        let index =
            Index::from_vectors_and_items(deduplicated_embeddings, items, options.ann_backend)?;

        Ok(Self {
            index,
            model,
            columns,
            was_string,
            ranking_cache: Mutex::new(None),
        })
    }

    /// Perform cross-dataset deduplication against the fitted index.
    pub fn deduplicate(&self, records: Vec<Record>, threshold: f32) -> Result<DeduplicationResult> {
        let dict_records = validate_if_strings(&records, &self.columns, self.was_string)?;
        let (dict_records, exact_duplicates) = remove_exact_duplicates(
            &dict_records,
            &self.columns,
            Some(self.index.items.as_slice()),
        )?;

        let mut duplicate_records = Vec::new();
        for (record, duplicates) in exact_duplicates {
            duplicate_records.push(DuplicateRecord::new(
                Record::Dict(record),
                true,
                scored_dicts_to_records(add_scores_to_records(&duplicates)),
            ));
        }

        if dict_records.is_empty() {
            let result = DeduplicationResult::new(
                Vec::new(),
                duplicate_records,
                threshold,
                Some(self.columns.clone()),
            );
            return Ok(if self.was_string {
                map_deduplication_result_to_strings(result, &self.columns)
            } else {
                result
            });
        }

        let embeddings = featurize(&dict_records, &self.columns, self.model.as_ref())?;
        let results = self.index.query_threshold(&embeddings, threshold)?;
        let mut deduplicated_records = Vec::new();

        for (record, similar_items) in dict_records.into_iter().zip(results) {
            if similar_items.is_empty() {
                deduplicated_records.push(Record::Dict(record));
            } else {
                duplicate_records.push(DuplicateRecord::new(
                    Record::Dict(record),
                    false,
                    scored_dicts_to_records(similar_items),
                ));
            }
        }

        let result = DeduplicationResult::new(
            deduplicated_records,
            duplicate_records,
            threshold,
            Some(self.columns.clone()),
        );

        Ok(if self.was_string {
            map_deduplication_result_to_strings(result, &self.columns)
        } else {
            result
        })
    }

    /// Deduplicate within the fitted dataset.
    pub fn self_deduplicate(&self, threshold: f32) -> Result<DeduplicationResult> {
        let results = self.index.query_threshold(&self.index.vectors, threshold)?;
        let mut duplicate_records = Vec::new();
        let mut deduplicated_records = Vec::new();
        let mut seen_items: HashSet<std::collections::BTreeMap<String, Value>> = HashSet::new();

        for (item, similar_items) in self.index.items.iter().zip(results) {
            if item.is_empty() {
                continue;
            }
            let record = item[0].clone();
            let duplicates = &item[1..];

            for (index, curr_record) in duplicates.iter().enumerate() {
                let actual_index = index + 1;
                let mut items_to_keep = Vec::new();
                items_to_keep.extend_from_slice(&item[..actual_index]);
                items_to_keep.extend_from_slice(&item[actual_index + 1..]);
                duplicate_records.push(DuplicateRecord::new(
                    Record::Dict(curr_record.clone()),
                    true,
                    scored_dicts_to_records(add_scores_to_records(&items_to_keep)),
                ));
            }

            if similar_items.is_empty() {
                deduplicated_records.push(Record::Dict(record));
                continue;
            }

            let mut frozen_items = Vec::with_capacity(similar_items.len());
            for (similar_item, _) in &similar_items {
                frozen_items.push(to_key_map(similar_item, &self.columns)?);
            }

            if frozen_items.iter().any(|item| seen_items.contains(item)) {
                let duplicates = similar_items
                    .into_iter()
                    .filter(|(item, _)| item != &record)
                    .collect();
                duplicate_records.push(DuplicateRecord::new(
                    Record::Dict(record),
                    false,
                    scored_dicts_to_records(duplicates),
                ));
                continue;
            }

            deduplicated_records.push(Record::Dict(record));
            seen_items.extend(frozen_items);
        }

        let result = DeduplicationResult::new(
            deduplicated_records,
            duplicate_records,
            threshold,
            Some(self.columns.clone()),
        );

        Ok(if self.was_string {
            map_deduplication_result_to_strings(result, &self.columns)
        } else {
            result
        })
    }

    /// Find representative samples from a given set of records against the
    /// fitted index.
    pub fn find_representative(
        &self,
        records: Vec<Record>,
        selection_size: usize,
        candidate_limit: CandidateLimit,
        diversity: f32,
        strategy: Strategy,
    ) -> Result<FilterResult> {
        let ranking = self.rank_by_average_similarity(records)?;
        let candidate_limit = match candidate_limit {
            CandidateLimit::Auto => {
                compute_candidate_limit_default(ranking.selected.len(), selection_size)
            }
            CandidateLimit::Value(value) => value,
        };
        self.diversify_ranked(
            ranking,
            candidate_limit,
            selection_size,
            diversity,
            strategy,
        )
    }

    /// Find representative samples from the fitted dataset.
    pub fn self_find_representative(
        &self,
        selection_size: usize,
        candidate_limit: CandidateLimit,
        diversity: f32,
        strategy: Strategy,
    ) -> Result<FilterResult> {
        let ranking = self.self_rank_by_average_similarity()?;
        let candidate_limit = match candidate_limit {
            CandidateLimit::Auto => {
                compute_candidate_limit_default(ranking.selected.len(), selection_size)
            }
            CandidateLimit::Value(value) => value,
        };
        self.diversify_ranked(
            ranking,
            candidate_limit,
            selection_size,
            diversity,
            strategy,
        )
    }

    /// Filter outliers in a given set of records against the fitted dataset.
    pub fn filter_outliers(
        &self,
        records: Vec<Record>,
        outlier_percentage: f32,
    ) -> Result<FilterResult> {
        if !(0.0..=1.0).contains(&outlier_percentage) {
            return Err(SemHashError::InvalidPercentage {
                name: "outlier_percentage",
            });
        }
        let ranking = self.rank_by_average_similarity(records)?;
        split_outliers(ranking, outlier_percentage)
    }

    /// Filter outliers in the fitted dataset.
    pub fn self_filter_outliers(&self, outlier_percentage: f32) -> Result<FilterResult> {
        if !(0.0..=1.0).contains(&outlier_percentage) {
            return Err(SemHashError::InvalidPercentage {
                name: "outlier_percentage",
            });
        }
        let ranking = self.self_rank_by_average_similarity()?;
        split_outliers(ranking, outlier_percentage)
    }

    /// Rank records based on average cosine similarity to neighbors in the
    /// fitted index.
    pub fn rank_by_average_similarity(&self, records: Vec<Record>) -> Result<FilterResult> {
        let dict_records = validate_if_strings(&records, &self.columns, self.was_string)?;
        let embeddings = featurize(&dict_records, &self.columns, self.model.as_ref())?;
        let scores = self.index.mean_top_k_similarity(&embeddings, 100, false)?;

        let mut sorted_scores: Vec<(usize, Record, f32)> = dict_records
            .into_iter()
            .zip(scores)
            .enumerate()
            .map(|(index, (record, score))| (index, Record::Dict(record), score))
            .collect();
        sorted_scores.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let (selected, scores_selected): (Vec<_>, Vec<_>) = sorted_scores
            .into_iter()
            .map(|(_, record, score)| (record, score))
            .unzip();
        Ok(FilterResult::new(
            selected,
            Vec::new(),
            scores_selected,
            Vec::new(),
        ))
    }

    /// Rank fitted index records by average neighbor similarity.
    pub fn self_rank_by_average_similarity(&self) -> Result<FilterResult> {
        if let Some(cached) = self
            .ranking_cache
            .lock()
            .expect("ranking cache poisoned")
            .clone()
        {
            return Ok(cached);
        }

        let dict_records: Vec<DictRecord> = self
            .index
            .items
            .iter()
            .filter_map(|bucket| bucket.first().cloned())
            .collect();
        let scores = self
            .index
            .mean_top_k_similarity(&self.index.vectors, 100, true)?;

        let mut sorted_scores: Vec<(usize, Record, f32)> = dict_records
            .into_iter()
            .zip(scores)
            .enumerate()
            .map(|(index, (record, score))| (index, Record::Dict(record), score))
            .collect();
        sorted_scores.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let (selected, scores_selected): (Vec<_>, Vec<_>) = sorted_scores
            .into_iter()
            .map(|(_, record, score)| (record, score))
            .unzip();
        let ranking = FilterResult::new(selected, Vec::new(), scores_selected, Vec::new());
        *self.ranking_cache.lock().expect("ranking cache poisoned") = Some(ranking.clone());
        Ok(ranking)
    }

    /// Diversify top candidates using the specified strategy.
    pub fn diversify_ranked(
        &self,
        ranked_results: FilterResult,
        candidate_limit: usize,
        selection_size: usize,
        diversity: f32,
        strategy: Strategy,
    ) -> Result<FilterResult> {
        let candidate_limit = candidate_limit.min(ranked_results.selected.len());
        let candidates = ranked_results.selected[..candidate_limit].to_vec();
        let relevance = ranked_results.scores_selected[..candidate_limit].to_vec();

        if candidates.is_empty() {
            return Ok(FilterResult::new(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
        }

        let candidate_dicts = records_to_dicts_for_featurize(&candidates, &self.columns)?;
        let embeddings = featurize(&candidate_dicts, &self.columns, self.model.as_ref())?;
        let result = diversify(&embeddings, &relevance, selection_size, strategy, diversity)?;
        let selected_set: HashSet<usize> = result.indices.iter().copied().collect();

        let selected = result
            .indices
            .iter()
            .map(|index| candidates[*index].clone())
            .collect();
        let filtered = candidates
            .into_iter()
            .enumerate()
            .filter_map(|(index, record)| (!selected_set.contains(&index)).then_some(record))
            .collect();
        let scores_filtered = relevance
            .iter()
            .enumerate()
            .filter_map(|(index, score)| (!selected_set.contains(&index)).then_some(*score))
            .collect();

        Ok(FilterResult::new(
            selected,
            filtered,
            result.selection_scores,
            scores_filtered,
        ))
    }

    /// Whether the instance was built from text records.
    pub fn was_string(&self) -> bool {
        self.was_string
    }
}

fn split_outliers(ranking: FilterResult, outlier_percentage: f32) -> Result<FilterResult> {
    let outlier_count = ((ranking.selected.len() as f32) * outlier_percentage).ceil() as usize;
    if outlier_count == 0 {
        return Ok(FilterResult::new(
            ranking.selected,
            Vec::new(),
            ranking.scores_selected,
            Vec::new(),
        ));
    }

    let split_at = ranking.selected.len().saturating_sub(outlier_count);
    let selected = ranking.selected[..split_at].to_vec();
    let filtered = ranking.selected[split_at..].to_vec();
    let scores_selected = ranking.scores_selected[..split_at].to_vec();
    let scores_filtered = ranking.scores_selected[split_at..].to_vec();
    Ok(FilterResult::new(
        selected,
        filtered,
        scores_selected,
        scores_filtered,
    ))
}

#[allow(dead_code)]
fn _dicts_to_records_for_public(records: Vec<DictRecord>) -> Vec<Record> {
    dicts_to_records(records)
}
