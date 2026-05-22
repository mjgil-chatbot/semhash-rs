import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CASES = (
    (1000, 300, 48, 96, 8),
    (2500, 800, 64, 128, 5),
    (5000, 1500, 96, 160, 3),
)

ORDER_INSENSITIVE_KEYS = {
    "self_outliers_filtered",
    "cross_outliers_filtered",
    "self_representative_selected",
    "cross_representative_selected",
}


@dataclass(frozen=True)
class Case:
    train: int
    test: int
    clusters: int
    dimensions: int
    repeats: int


class DeterministicEncoder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = max(dimensions, 8)

    def encode(self, inputs):
        import numpy as np

        return np.array([encode_text(str(value), self.dimensions) for value in inputs], dtype="float32")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-repo",
        default="/home/m/git/others/semhash",
        help="Path to the Python semhash reference repository.",
    )
    parser.add_argument(
        "--rust-binary",
        default="target/release/examples/reference_benchmark",
        help="Path to the compiled Rust benchmark example.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case in train,test,clusters,dimensions,repeats form. May be passed multiple times.",
    )
    args = parser.parse_args()

    reference_repo = Path(args.reference_repo).resolve()
    rust_binary = Path(args.rust_binary).resolve()
    cases = [parse_case(case) for case in args.case] or [Case(*case) for case in DEFAULT_CASES]

    if not rust_binary.exists():
        raise SystemExit(f"missing Rust benchmark binary: {rust_binary}")

    sys.path.insert(0, str(reference_repo))
    from semhash import SemHash, Strategy
    from vicinity import Backend

    print(
        "| Case | Build | Self Dedup | Cross Dedup | Self Outliers | Cross Outliers | Self Repr | Cross Repr |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    for case in cases:
        rust_result = run_rust_case(rust_binary, case)
        python_result = run_python_case(reference_repo, case, SemHash, Strategy, Backend)
        compare_outputs(case, rust_result["outputs"], python_result["outputs"])
        print(format_row(case, rust_result["timings_ms"], python_result["timings_ms"]), flush=True)

    return 0


def parse_case(value: str) -> Case:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 5:
        raise ValueError(f"invalid case: {value}")
    return Case(*parts)


def run_rust_case(rust_binary: Path, case: Case) -> dict:
    proc = subprocess.run(
        [
            str(rust_binary),
            str(case.train),
            str(case.test),
            str(case.clusters),
            str(case.dimensions),
            str(case.repeats),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def run_python_case(reference_repo: Path, case: Case, SemHash, Strategy, Backend) -> dict:
    train_records = [train_text(idx, case.clusters) for idx in range(case.train)]
    test_records = [test_text(idx, case.train, case.clusters) for idx in range(case.test)]

    timings = {
        "build": [],
        "self_deduplicate": [],
        "deduplicate": [],
        "self_filter_outliers": [],
        "filter_outliers": [],
        "self_find_representative": [],
        "find_representative": [],
    }
    output = None

    for _ in range(case.repeats):
        encoder = DeterministicEncoder(case.dimensions)

        build_start = time.perf_counter()
        semhash = SemHash.from_records(
            records=train_records,
            model=encoder,
            ann_backend=Backend.BASIC,
        )
        timings["build"].append(elapsed_ms(build_start))

        self_dedup_start = time.perf_counter()
        self_dedup = semhash.self_deduplicate(threshold=0.92)
        timings["self_deduplicate"].append(elapsed_ms(self_dedup_start))

        cross_dedup_start = time.perf_counter()
        cross_dedup = semhash.deduplicate(test_records, threshold=0.92)
        timings["deduplicate"].append(elapsed_ms(cross_dedup_start))

        self_outliers_start = time.perf_counter()
        self_outliers = semhash.self_filter_outliers(outlier_percentage=0.10)
        timings["self_filter_outliers"].append(elapsed_ms(self_outliers_start))

        cross_outliers_start = time.perf_counter()
        cross_outliers = semhash.filter_outliers(test_records, outlier_percentage=0.10)
        timings["filter_outliers"].append(elapsed_ms(cross_outliers_start))

        self_representative_start = time.perf_counter()
        self_representative = semhash.self_find_representative(
            selection_size=12,
            candidate_limit=case.train,
            diversity=0.50,
            strategy=Strategy.MMR,
        )
        timings["self_find_representative"].append(elapsed_ms(self_representative_start))

        cross_representative_start = time.perf_counter()
        cross_representative = semhash.find_representative(
            records=test_records,
            selection_size=12,
            candidate_limit=case.test,
            diversity=0.50,
            strategy=Strategy.MMR,
        )
        timings["find_representative"].append(elapsed_ms(cross_representative_start))

        if output is None:
            output = {
                "self_dedup_selected": list(self_dedup.selected),
                "self_dedup_filtered": [dup.record for dup in self_dedup.filtered],
                "cross_dedup_selected": list(cross_dedup.selected),
                "cross_dedup_filtered": [dup.record for dup in cross_dedup.filtered],
                "self_outliers_filtered": [record["text"] for record in self_outliers.filtered],
                "cross_outliers_filtered": [record["text"] for record in cross_outliers.filtered],
                "self_representative_selected": [record["text"] for record in self_representative.selected],
                "cross_representative_selected": [record["text"] for record in cross_representative.selected],
            }

    return {
        "timings_ms": {key: statistics.mean(values) for key, values in timings.items()},
        "outputs": output,
    }


def compare_outputs(case: Case, rust_outputs: dict, python_outputs: dict) -> None:
    for key in sorted(rust_outputs):
        rust_value = rust_outputs[key]
        python_value = python_outputs[key]
        if key in ORDER_INSENSITIVE_KEYS:
            rust_value = sorted(rust_value)
            python_value = sorted(python_value)
        if rust_value != python_value:
            raise AssertionError(
                f"output mismatch for case {case} and key {key}\n"
                f"rust:   {rust_value}\n"
                f"python: {python_value}"
            )


def format_row(case: Case, rust: dict, python: dict) -> str:
    label = f"{case.train}/{case.test}/{case.clusters}/{case.dimensions}x{case.repeats}"
    return (
        f"| `{label}` "
        f"| {ratio(rust['build'], python['build'])} "
        f"| {ratio(rust['self_deduplicate'], python['self_deduplicate'])} "
        f"| {ratio(rust['deduplicate'], python['deduplicate'])} "
        f"| {ratio(rust['self_filter_outliers'], python['self_filter_outliers'])} "
        f"| {ratio(rust['filter_outliers'], python['filter_outliers'])} "
        f"| {ratio(rust['self_find_representative'], python['self_find_representative'])} "
        f"| {ratio(rust['find_representative'], python['find_representative'])} |"
    )


def ratio(rust_ms: float, python_ms: float) -> str:
    if rust_ms == 0 or python_ms == 0:
        return f"`rust {rust_ms:.2f}ms / py {python_ms:.2f}ms`"
    speedup = python_ms / rust_ms
    return f"`rust {rust_ms:.2f}ms / py {python_ms:.2f}ms / {speedup:.2f}x`"


def train_text(index: int, clusters: int) -> str:
    if index > 0 and index % 11 == 0:
        return train_text(index - 1, clusters)
    if index % 23 == 0:
        cluster = clusters + (index % 7)
        variant = index // 23
        return f"kind:outlier cluster:{cluster} variant:{variant}"
    cluster = index % clusters
    variant = index
    return f"kind:train cluster:{cluster} variant:{variant}"


def test_text(index: int, train: int, clusters: int) -> str:
    if index % 9 == 0:
        source = (index * 13) % max(train, 1)
        return train_text(source, clusters)
    if index % 17 == 0:
        cluster = clusters + (index % 5)
        variant = train + index
        return f"kind:outlier cluster:{cluster} variant:{variant}"
    cluster = index % clusters
    variant = train + index
    return f"kind:test cluster:{cluster} variant:{variant}"


def encode_text(text: str, dimensions: int) -> list[float]:
    cluster = parse_field(text, "cluster") or 0
    variant = parse_field(text, "variant") or 0
    outlier = "kind:outlier" in text

    vector = [0.0] * dimensions
    base = cluster % dimensions
    vector[base] = 0.4 if outlier else 1.0

    secondary = (cluster * 7 + variant) % dimensions
    vector[secondary] += 0.9 if outlier else 0.25

    tertiary = (cluster * 13 + variant // 7 + 3) % dimensions
    vector[tertiary] += 0.2 if outlier else 0.05

    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def parse_field(text: str, name: str) -> int | None:
    for part in text.split():
        key, _, value = part.partition(":")
        if key == name:
            return int(value)
    return None


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
