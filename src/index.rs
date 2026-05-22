use crate::error::{Result, SemHashError};
use crate::utils::{normalize, validate_embeddings_shape, DictRecord, Embeddings};
use ndarray::{linalg::general_mat_mul, Array2, ArrayView2};
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
    normalized_vectors: Vec<f32>,
    dimensions: usize,
}

impl Index {
    pub fn new(vectors: Embeddings, items: Vec<Vec<DictRecord>>, backend: Backend) -> Result<Self> {
        let dimensions = validate_embeddings_shape(&vectors, items.len(), None)?;
        if vectors.len() != items.len() {
            return Err(SemHashError::InvalidEmbeddings(format!(
                "Number of vectors ({}) must match number of items ({})",
                vectors.len(),
                items.len()
            )));
        }

        Ok(Self {
            normalized_vectors: flatten_normalized_vectors(&vectors, dimensions),
            vectors,
            items,
            backend,
            dimensions,
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
        self.similarity_batches(vectors, |_, row| {
            let mut neighbors: Vec<(usize, f32)> = row
                .iter()
                .copied()
                .enumerate()
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
    }

    /// Query the index with top-k nearest neighbors.
    pub fn query_top_k(
        &self,
        vectors: &Embeddings,
        k: usize,
        vectors_are_in_index: bool,
    ) -> Result<Vec<SingleQueryResult>> {
        self.similarity_batches(vectors, |query_index, row| {
            let mut neighbors: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            if vectors_are_in_index && query_index < neighbors.len() {
                neighbors.swap_remove(query_index);
            }

            let take = k.min(neighbors.len());
            trim_top_k(&mut neighbors, take);
            let indices = neighbors.iter().map(|(index, _)| *index).collect();
            let similarities = neighbors
                .iter()
                .map(|(_, similarity)| *similarity)
                .collect();
            (indices, similarities)
        })
    }

    /// Compute the mean similarity of the top-k nearest neighbors for each
    /// query vector.
    pub fn mean_top_k_similarity(
        &self,
        vectors: &Embeddings,
        k: usize,
        vectors_are_in_index: bool,
    ) -> Result<Vec<f32>> {
        let take = k.min(
            self.vectors
                .len()
                .saturating_sub(usize::from(vectors_are_in_index)),
        );

        self.similarity_batches(vectors, |query_index, row| {
            if take == 0 {
                return 0.0;
            }

            let mut best = Vec::with_capacity(take);
            for (index, similarity) in row.iter().copied().enumerate() {
                if vectors_are_in_index && index == query_index {
                    continue;
                }
                push_top_similarity(&mut best, similarity, take);
            }

            if best.is_empty() {
                0.0
            } else {
                best.iter().copied().sum::<f32>() / best.len() as f32
            }
        })
    }

    fn similarity_batches<T, F>(&self, vectors: &Embeddings, project: F) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(usize, &[f32]) -> T + Send + Sync,
    {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        validate_query_dims(vectors, self.dimensions)?;

        let corpus = ArrayView2::from_shape(
            (self.vectors.len(), self.dimensions),
            self.normalized_vectors.as_slice(),
        )
        .expect("normalized corpus shape should be valid");

        const QUERY_BATCH_SIZE: usize = 128;

        let batches: Vec<Vec<T>> = vectors
            .par_chunks(QUERY_BATCH_SIZE)
            .enumerate()
            .map(|(batch_index, chunk)| {
                let start_index = batch_index * QUERY_BATCH_SIZE;
                let query_flat = flatten_normalized_vectors(chunk, self.dimensions);
                let query_matrix =
                    ArrayView2::from_shape((chunk.len(), self.dimensions), query_flat.as_slice())
                        .expect("normalized query shape should be valid");

                let mut similarities = Array2::<f32>::zeros((chunk.len(), self.vectors.len()));
                general_mat_mul(1.0, &query_matrix, &corpus.t(), 0.0, &mut similarities);

                similarities
                    .outer_iter()
                    .enumerate()
                    .map(|(local_index, row)| {
                        let row = row
                            .as_slice()
                            .expect("similarity rows should be contiguous");
                        project(start_index + local_index, row)
                    })
                    .collect()
            })
            .collect();

        Ok(batches.into_iter().flatten().collect())
    }
}

fn flatten_normalized_vectors(vectors: &[Vec<f32>], dimensions: usize) -> Vec<f32> {
    let mut flat = Vec::with_capacity(vectors.len() * dimensions);
    for vector in vectors {
        let mut normalized = vector.clone();
        normalize(&mut normalized);
        flat.extend(normalized);
    }
    flat
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
