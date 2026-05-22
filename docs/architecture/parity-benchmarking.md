# Parity Benchmarking

This repository now carries a dedicated cross-repo parity harness for the upstream Python `semhash` reference implementation.

## Scope

- `tests/reference_parity.rs` pins reference API behavior that matters for this port:
  - edge-case validation and error messages for `deduplicate`
  - cached `selected_with_duplicates` invalidation after `rethreshold`
  - empty `FilterResult` ratios
  - representative selection and outlier filtering oracles for a deterministic encoder
- `examples/reference_benchmark.rs` is the Rust-side benchmark target.
- `scripts/benchmark_reference_parity.py` orchestrates a direct comparison against the Python reference repo.

## Implementation Notes

- Ranking sorts in `src/semhash.rs` break score ties by original input position so they match Python's stable ordering semantics.
- Exact-search queries in `src/index.rs` are parallelized with `rayon` at the query level.
- Top-k retrieval now trims to the requested prefix before sorting it, instead of fully sorting every candidate list.
- Ranking-based outlier and representative flows now use a mean-top-k similarity path, avoiding neighbor-id materialization when only aggregate scores are needed.

## Benchmark Design

The benchmark intentionally does not compare the Python default runtime configuration against the Rust default runtime configuration:

- Python upstream defaults to `Backend.USEARCH`, which is approximate.
- This Rust port currently routes all backends through exact cosine search.
- Comparing those defaults would mix implementation differences with ANN backend differences.

Instead, both sides are run with:

- a deterministic encoder implemented independently in Rust and Python
- exact search backends (`Backend.BASIC` in Python, `Backend::Exact` in Rust)
- the same synthetic text workloads, duplicate patterns, thresholds, outlier rates, and representative-selection settings
- explicit `candidate_limit` values for representative selection so benchmark outcomes do not drift on top-k boundary ties

That makes the harness useful for two purposes:

- confirming API-output parity on the same workloads
- measuring speed deltas on comparable algorithmic work

## Workload Ownership

The current parity harness covers the text-mode path of the crate:

- `SemHash::from_records`
- `self_deduplicate`
- `deduplicate`
- `self_filter_outliers`
- `filter_outliers`
- `self_find_representative`
- `find_representative`

Dictionary-record parity continues to be covered in the Rust integration tests because the benchmark workload is focused on stable, repeatable speed comparisons rather than exhaustive public-surface enumeration.

## Running

1. Build the Rust benchmark example in release mode.
2. Run `scripts/benchmark_reference_parity.py` and point it at the Python reference checkout if needed.

The script rejects the run if any benchmark scenario produces divergent outputs between the two implementations.

For ranking-derived outputs (`filter_outliers` and representative selection), the script compares selected membership rather than exact list order in the benchmark harness. The Rust integration tests keep explicit oracle assertions for deterministic cases, while the cross-repo benchmark avoids failing on backend-level tie ordering that does not change the selected records.
