use crate::error::{Result, SemHashError};
use crate::utils::{cosine_similarity, validate_embeddings_shape, DictRecord, Embeddings};
use rayon::prelude::*;

/// Backend selector matching the Python `ann_backend` parameter.
///
/// This port keeps the backend parameter for API compatibility. The included
/// implementation is an exact cosine backend; all variants route to exact
/// search unless a downstream fork wires an ANN backend behind the enum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    #[default]
    Usearch,
    Exact,
    Hnsw,
    Flat,
}

/// Top-k query result: index ids and cosine similarity scores.
pub type SingleQueryResult = (Vec<usize>, Vec<f32>);

/// Threshold query document score.
pub type DocScore = (DictRecord, f32);

/// Threshold query scores for one query.
pub type DocScores = Vec<DocScore>;

/// A vector index that maps vectors to exact-duplicate record buckets.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    pub vectors: Embeddings,
    pub items: Vec<Vec<DictRecord>>,
    pub backend: Backend,
}

impl Index {
    pub fn new(vectors: Embeddings, items: Vec<Vec<DictRecord>>, backend: Backend) -> Result<Self> {
        validate_embeddings_shape(&vectors, items.len(), None)?;
        if vectors.len() != items.len() {
            return Err(SemHashError::InvalidEmbeddings(format!(
                "Number of vectors ({}) must match number of items ({})",
                vectors.len(),
                items.len()
            )));
        }
        Ok(Self {
            vectors,
            items,
            backend,
        })
    }

    /// Load the index from vectors and items.
    pub fn from_vectors_and_items(
        vectors: Embeddings,
        items: Vec<Vec<DictRecord>>,
        backend_type: Backend,
    ) -> Result<Self> {
        Self::new(vectors, items, backend_type)
    }

    /// Query the index with a cosine-similarity threshold.
    ///
    /// Python Vicinity returns distances and SemHash converts them back with
    /// `1 - distance`. This exact backend directly computes cosine similarity.
    pub fn query_threshold(&self, vectors: &Embeddings, threshold: f32) -> Result<Vec<DocScores>> {
        if self.vectors.is_empty() {
            return Ok(vec![Vec::new(); vectors.len()]);
        }
        validate_query_dims(vectors, self.vectors[0].len())?;

        Ok(vectors
            .par_iter()
            .map(|query| {
                let mut neighbors: Vec<(usize, f32)> = self
                    .vectors
                    .iter()
                    .enumerate()
                    .map(|(index, vector)| (index, cosine_similarity(query, vector)))
                    .filter(|(_, similarity)| *similarity >= threshold)
                    .collect();
                trim_top_k(&mut neighbors, 100);

                let mut intermediate = Vec::new();
                for (index, similarity) in neighbors {
                    for record in &self.items[index] {
                        intermediate.push((record.clone(), similarity));
                    }
                }
                intermediate
            })
            .collect())
    }

    /// Query the index with top-k nearest neighbors.
    pub fn query_top_k(
        &self,
        vectors: &Embeddings,
        k: usize,
        vectors_are_in_index: bool,
    ) -> Result<Vec<SingleQueryResult>> {
        if self.vectors.is_empty() {
            return Ok(vec![(Vec::new(), Vec::new()); vectors.len()]);
        }
        validate_query_dims(vectors, self.vectors[0].len())?;

        let offset = usize::from(vectors_are_in_index);
        Ok(vectors
            .par_iter()
            .map(|query| {
                let mut neighbors: Vec<(usize, f32)> = self
                    .vectors
                    .iter()
                    .enumerate()
                    .map(|(index, vector)| (index, cosine_similarity(query, vector)))
                    .collect();
                let take = k.saturating_add(offset).min(neighbors.len());
                trim_top_k(&mut neighbors, take);
                let sliced = if offset > 0 && take > 0 {
                    &neighbors[offset..take]
                } else {
                    &neighbors[..take]
                };
                let indices = sliced.iter().map(|(index, _)| *index).collect();
                let similarities = sliced.iter().map(|(_, similarity)| *similarity).collect();
                (indices, similarities)
            })
            .collect())
    }

    /// Compute the mean similarity of the top-k nearest neighbors for each
    /// query vector.
    pub fn mean_top_k_similarity(
        &self,
        vectors: &Embeddings,
        k: usize,
        vectors_are_in_index: bool,
    ) -> Result<Vec<f32>> {
        if self.vectors.is_empty() {
            return Ok(vec![0.0; vectors.len()]);
        }
        validate_query_dims(vectors, self.vectors[0].len())?;

        let offset = usize::from(vectors_are_in_index);
        let take = k.min(self.vectors.len().saturating_sub(offset));

        Ok(vectors
            .par_iter()
            .enumerate()
            .map(|(query_index, query)| {
                if take == 0 {
                    return 0.0;
                }

                let mut best = Vec::with_capacity(take);
                for (index, vector) in self.vectors.iter().enumerate() {
                    if vectors_are_in_index && index == query_index {
                        continue;
                    }
                    let similarity = cosine_similarity(query, vector);
                    push_top_similarity(&mut best, similarity, take);
                }

                if best.is_empty() {
                    0.0
                } else {
                    best.iter().copied().sum::<f32>() / best.len() as f32
                }
            })
            .collect())
    }
}

fn validate_query_dims(vectors: &Embeddings, expected_cols: usize) -> Result<()> {
    validate_embeddings_shape(vectors, vectors.len(), Some(expected_cols)).map(|_| ())
}

fn trim_top_k(neighbors: &mut Vec<(usize, f32)>, take: usize) {
    if take == 0 {
        neighbors.clear();
        return;
    }

    if neighbors.len() > take {
        neighbors.select_nth_unstable_by(take, compare_neighbors);
        neighbors.truncate(take);
    }
    neighbors.sort_unstable_by(compare_neighbors);
}

fn compare_neighbors(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

fn push_top_similarity(best: &mut Vec<f32>, similarity: f32, take: usize) {
    if best.len() < take {
        let insert_at = best.partition_point(|value| *value < similarity);
        best.insert(insert_at, similarity);
        return;
    }

    if similarity <= best[0] {
        return;
    }

    best.remove(0);
    let insert_at = best.partition_point(|value| *value < similarity);
    best.insert(insert_at, similarity);
}
