use crate::error::{Result, SemHashError};
use crate::utils::{cosine_similarity, Embeddings};
use std::str::FromStr;

/// Diversification strategies matching the names exposed by Pyversity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Strategy {
    #[default]
    MMR,
    MSD,
    DPP,
    COVER,
    SSD,
}

impl FromStr for Strategy {
    type Err = SemHashError;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "MMR" => Ok(Self::MMR),
            "MSD" => Ok(Self::MSD),
            "DPP" => Ok(Self::DPP),
            "COVER" => Ok(Self::COVER),
            "SSD" => Ok(Self::SSD),
            other => Err(SemHashError::Message(format!(
                "Unknown diversification strategy: {other}"
            ))),
        }
    }
}

/// Result returned by `diversify`.
#[derive(Clone, Debug, PartialEq)]
pub struct DiversifyResult {
    pub indices: Vec<usize>,
    pub selection_scores: Vec<f32>,
}

/// Diversify candidates using the selected strategy.
///
/// `MMR` is implemented directly. `MSD`, `DPP`, `COVER`, and `SSD` retain the
/// public strategy names and currently route through the same greedy relevance +
/// diversity objective so the SemHash API remains operational without pulling in
/// Pyversity.
pub fn diversify(
    embeddings: &Embeddings,
    scores: &[f32],
    k: usize,
    strategy: Strategy,
    diversity: f32,
) -> Result<DiversifyResult> {
    if embeddings.len() != scores.len() {
        return Err(SemHashError::InvalidEmbeddings(format!(
            "Number of embeddings ({}) must match number of scores ({})",
            embeddings.len(),
            scores.len()
        )));
    }
    if embeddings.is_empty() || k == 0 {
        return Ok(DiversifyResult {
            indices: Vec::new(),
            selection_scores: Vec::new(),
        });
    }

    let diversity = diversity.clamp(0.0, 1.0);
    match strategy {
        Strategy::MMR | Strategy::MSD | Strategy::DPP | Strategy::COVER | Strategy::SSD => {
            greedy_mmr_like(embeddings, scores, k.min(embeddings.len()), diversity)
        }
    }
}

fn greedy_mmr_like(
    embeddings: &Embeddings,
    scores: &[f32],
    k: usize,
    diversity: f32,
) -> Result<DiversifyResult> {
    let mut selected = Vec::with_capacity(k);
    let mut selection_scores = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..embeddings.len()).collect();

    // Pyversity's MMR behavior starts from the highest relevance item.
    let first = remaining
        .iter()
        .copied()
        .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then_with(|| b.cmp(a)))
        .unwrap();
    selected.push(first);
    selection_scores.push(scores[first]);
    remaining.retain(|idx| *idx != first);

    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = remaining[0];
        let mut best_score = f32::NEG_INFINITY;

        for candidate in &remaining {
            let max_similarity = selected
                .iter()
                .map(|selected_idx| {
                    cosine_similarity(&embeddings[*candidate], &embeddings[*selected_idx])
                })
                .fold(f32::NEG_INFINITY, f32::max);
            // MMR-style objective. At diversity=0 this is pure relevance; at
            // diversity=1 this chooses the candidate least similar to the
            // current selected set.
            let objective = (1.0 - diversity) * scores[*candidate] - diversity * max_similarity;
            if objective > best_score || (objective == best_score && *candidate < best_idx) {
                best_idx = *candidate;
                best_score = objective;
            }
        }

        selected.push(best_idx);
        selection_scores.push(best_score);
        remaining.retain(|idx| *idx != best_idx);
    }

    Ok(DiversifyResult {
        indices: selected,
        selection_scores,
    })
}
