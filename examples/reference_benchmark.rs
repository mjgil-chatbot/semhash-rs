use semhash_rs::{
    Backend, CandidateLimit, Embeddings, Encoder, Record, Result, SemHash, SemHashOptions,
    Strategy, Value,
};
use std::env;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Config {
    train: usize,
    test: usize,
    clusters: usize,
    dimensions: usize,
    repeats: usize,
    threshold: f32,
    outlier_percentage: f32,
    selection_size: usize,
    diversity: f32,
}

#[derive(Default)]
struct Timings {
    build_ms: f64,
    self_dedup_ms: f64,
    cross_dedup_ms: f64,
    self_outliers_ms: f64,
    cross_outliers_ms: f64,
    self_representative_ms: f64,
    cross_representative_ms: f64,
}

#[derive(Clone, Debug)]
struct DeterministicEncoder {
    dimensions: usize,
}

impl DeterministicEncoder {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Encoder for DeterministicEncoder {
    fn encode(&self, inputs: &[Value]) -> Result<Embeddings> {
        Ok(inputs
            .iter()
            .map(|value| encode_text(&value.as_string_lossy(), self.dimensions))
            .collect())
    }
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1));
    let output = benchmark(config)?;
    println!("{output}");
    Ok(())
}

fn benchmark(config: Config) -> Result<String> {
    let (train_records, test_records) =
        generate_records(config.train, config.test, config.clusters);
    let mut timings = Timings::default();
    let mut first_result: Option<ScenarioOutput> = None;

    for _ in 0..config.repeats {
        let encoder = Arc::new(DeterministicEncoder::new(config.dimensions));

        let build_start = Instant::now();
        let semhash = SemHash::from_records(
            train_records.clone(),
            SemHashOptions::default()
                .model(encoder)
                .ann_backend(Backend::Exact),
        )?;
        timings.build_ms += elapsed_ms(build_start);

        let self_dedup_start = Instant::now();
        let self_dedup = semhash.self_deduplicate(config.threshold)?;
        timings.self_dedup_ms += elapsed_ms(self_dedup_start);

        let cross_dedup_start = Instant::now();
        let cross_dedup = semhash.deduplicate(test_records.clone(), config.threshold)?;
        timings.cross_dedup_ms += elapsed_ms(cross_dedup_start);

        let self_outliers_start = Instant::now();
        let self_outliers = semhash.self_filter_outliers(config.outlier_percentage)?;
        timings.self_outliers_ms += elapsed_ms(self_outliers_start);

        let cross_outliers_start = Instant::now();
        let cross_outliers =
            semhash.filter_outliers(test_records.clone(), config.outlier_percentage)?;
        timings.cross_outliers_ms += elapsed_ms(cross_outliers_start);

        let self_representative_start = Instant::now();
        let self_representative = semhash.self_find_representative(
            config.selection_size,
            CandidateLimit::Value(config.train),
            config.diversity,
            Strategy::MMR,
        )?;
        timings.self_representative_ms += elapsed_ms(self_representative_start);

        let cross_representative_start = Instant::now();
        let cross_representative = semhash.find_representative(
            test_records.clone(),
            config.selection_size,
            CandidateLimit::Value(config.test),
            config.diversity,
            Strategy::MMR,
        )?;
        timings.cross_representative_ms += elapsed_ms(cross_representative_start);

        if first_result.is_none() {
            first_result = Some(ScenarioOutput {
                self_dedup_selected: self_dedup.selected_texts().unwrap_or_default(),
                self_dedup_filtered: self_dedup
                    .filtered
                    .iter()
                    .map(|duplicate| record_text(&duplicate.record))
                    .collect(),
                cross_dedup_selected: cross_dedup.selected_texts().unwrap_or_default(),
                cross_dedup_filtered: cross_dedup
                    .filtered
                    .iter()
                    .map(|duplicate| record_text(&duplicate.record))
                    .collect(),
                self_outliers_filtered: self_outliers.filtered.iter().map(record_text).collect(),
                cross_outliers_filtered: cross_outliers.filtered.iter().map(record_text).collect(),
                self_representative_selected: self_representative
                    .selected
                    .iter()
                    .map(record_text)
                    .collect(),
                cross_representative_selected: cross_representative
                    .selected
                    .iter()
                    .map(record_text)
                    .collect(),
            });
        }
    }

    let repeats = config.repeats as f64;
    let output = first_result.expect("benchmark must run at least once");
    Ok(format!(
        "{{\"train_size\":{},\"test_size\":{},\"clusters\":{},\"dimensions\":{},\"repeats\":{},\"threshold\":{:.2},\"outlier_percentage\":{:.2},\"selection_size\":{},\"diversity\":{:.2},\"timings_ms\":{{\"build\":{:.4},\"self_deduplicate\":{:.4},\"deduplicate\":{:.4},\"self_filter_outliers\":{:.4},\"filter_outliers\":{:.4},\"self_find_representative\":{:.4},\"find_representative\":{:.4}}},\"outputs\":{{\"self_dedup_selected\":{},\"self_dedup_filtered\":{},\"cross_dedup_selected\":{},\"cross_dedup_filtered\":{},\"self_outliers_filtered\":{},\"cross_outliers_filtered\":{},\"self_representative_selected\":{},\"cross_representative_selected\":{}}}}}",
        config.train,
        config.test,
        config.clusters,
        config.dimensions,
        config.repeats,
        config.threshold,
        config.outlier_percentage,
        config.selection_size,
        config.diversity,
        timings.build_ms / repeats,
        timings.self_dedup_ms / repeats,
        timings.cross_dedup_ms / repeats,
        timings.self_outliers_ms / repeats,
        timings.cross_outliers_ms / repeats,
        timings.self_representative_ms / repeats,
        timings.cross_representative_ms / repeats,
        json_strings(&output.self_dedup_selected),
        json_strings(&output.self_dedup_filtered),
        json_strings(&output.cross_dedup_selected),
        json_strings(&output.cross_dedup_filtered),
        json_strings(&output.self_outliers_filtered),
        json_strings(&output.cross_outliers_filtered),
        json_strings(&output.self_representative_selected),
        json_strings(&output.cross_representative_selected),
    ))
}

struct ScenarioOutput {
    self_dedup_selected: Vec<String>,
    self_dedup_filtered: Vec<String>,
    cross_dedup_selected: Vec<String>,
    cross_dedup_filtered: Vec<String>,
    self_outliers_filtered: Vec<String>,
    cross_outliers_filtered: Vec<String>,
    self_representative_selected: Vec<String>,
    cross_representative_selected: Vec<String>,
}

fn parse_args<I>(args: I) -> Config
where
    I: IntoIterator<Item = String>,
{
    let mut values = args.into_iter();
    Config {
        train: parse_or_default(values.next(), 1_000),
        test: parse_or_default(values.next(), 300),
        clusters: parse_or_default(values.next(), 48),
        dimensions: parse_or_default(values.next(), 96),
        repeats: parse_or_default(values.next(), 5),
        threshold: 0.92,
        outlier_percentage: 0.10,
        selection_size: 12,
        diversity: 0.50,
    }
}

fn parse_or_default(value: Option<String>, default: usize) -> usize {
    value
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn generate_records(train: usize, test: usize, clusters: usize) -> (Vec<Record>, Vec<Record>) {
    let train_records = (0..train)
        .map(|idx| Record::from(train_text(idx, clusters)))
        .collect();
    let test_records = (0..test)
        .map(|idx| Record::from(test_text(idx, train, clusters)))
        .collect();
    (train_records, test_records)
}

fn train_text(index: usize, clusters: usize) -> String {
    if index > 0 && index % 11 == 0 {
        return train_text(index - 1, clusters);
    }
    if index % 23 == 0 {
        let cluster = clusters + (index % 7);
        let variant = index / 23;
        return format!("kind:outlier cluster:{cluster} variant:{variant}");
    }
    let cluster = index % clusters;
    let variant = index;
    format!("kind:train cluster:{cluster} variant:{variant}")
}

fn test_text(index: usize, train: usize, clusters: usize) -> String {
    if index % 9 == 0 {
        let source = (index * 13) % train.max(1);
        return train_text(source, clusters);
    }
    if index % 17 == 0 {
        let cluster = clusters + (index % 5);
        let variant = train + index;
        return format!("kind:outlier cluster:{cluster} variant:{variant}");
    }
    let cluster = index % clusters;
    let variant = train + index;
    format!("kind:test cluster:{cluster} variant:{variant}")
}

fn encode_text(text: &str, dimensions: usize) -> Vec<f32> {
    let dimensions = dimensions.max(8);
    let cluster = parse_field(text, "cluster").unwrap_or(0);
    let variant = parse_field(text, "variant").unwrap_or(0);
    let outlier = text.contains("kind:outlier");

    let mut vector = vec![0.0_f32; dimensions];
    let base = cluster % dimensions;
    vector[base] = if outlier { 0.4 } else { 1.0 };

    let secondary = (cluster * 7 + variant) % dimensions;
    vector[secondary] += if outlier { 0.9 } else { 0.25 };

    let tertiary = (cluster * 13 + variant / 7 + 3) % dimensions;
    vector[tertiary] += if outlier { 0.2 } else { 0.05 };

    normalize(&mut vector);
    vector
}

fn parse_field(text: &str, name: &str) -> Option<usize> {
    text.split_whitespace().find_map(|part| {
        let (key, value) = part.split_once(':')?;
        (key == name).then(|| value.parse().ok()).flatten()
    })
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn record_text(record: &Record) -> String {
    if let Some(text) = record.as_text() {
        text.to_string()
    } else {
        record
            .as_dict()
            .and_then(|record| record.get("text"))
            .map(Value::as_string_lossy)
            .expect("text record")
    }
}

fn json_strings(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", escape_json(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn escape_json(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
