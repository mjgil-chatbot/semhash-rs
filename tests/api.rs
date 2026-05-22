use semhash_rs::*;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
struct ToyEncoder;

impl Encoder for ToyEncoder {
    fn encode(&self, inputs: &[Value]) -> Result<Embeddings> {
        Ok(inputs.iter().map(embed_value).collect())
    }
}

fn embed_value(value: &Value) -> Vec<f32> {
    let text = value.as_string_lossy().to_lowercase();
    if [
        "apple",
        "banana",
        "cherry",
        "strawberry",
        "blueberry",
        "raspberry",
        "blackberry",
        "peach",
        "plum",
        "grape",
        "mango",
        "papaya",
        "pineapple",
        "watermelon",
        "orange",
        "lemon",
        "lime",
        "tangerine",
        "kiwi",
        "fig",
        "apricot",
        "grapefruit",
        "pomegranate",
    ]
    .contains(&text.as_str())
    {
        vec![1.0, 0.0, 0.0]
    } else if ["car", "bicycle", "motorcycle", "plane"].contains(&text.as_str()) {
        vec![0.0, 1.0, 0.0]
    } else if text.contains("dangerous") || text.contains("risky") || text.contains("safe") {
        vec![0.95, 0.05, 0.0]
    } else if text.contains("ganondorf") || text.contains("ganon") || text.contains("hyrule") {
        vec![0.0, 0.95, 0.05]
    } else if text.contains("link") || text.contains("hero") {
        vec![0.9, 0.1, 0.0]
    } else if text.contains("zelda") || text.contains("princess") {
        vec![0.1, 0.9, 0.0]
    } else {
        let encoder = HashingEncoder::new(3);
        encoder
            .encode(std::slice::from_ref(value))
            .unwrap()
            .remove(0)
    }
}

fn model() -> Arc<dyn Encoder> {
    Arc::new(ToyEncoder)
}

fn rec(pairs: &[(&str, Value)]) -> Record {
    let mut map = BTreeMap::new();
    for (key, value) in pairs {
        map.insert((*key).to_string(), value.clone());
    }
    Record::Dict(map)
}

fn text_records(values: &[&str]) -> Vec<Record> {
    values.iter().copied().map(Record::from).collect()
}

#[test]
fn deduplication_scoring_and_rethreshold() {
    let mut d = DeduplicationResult::new(
        vec![Record::from("a"), Record::from("b"), Record::from("c")],
        vec![
            DuplicateRecord::new(
                Record::from("d"),
                false,
                vec![(Record::from("x"), 0.9), (Record::from("y"), 0.8)],
            ),
            DuplicateRecord::new(Record::from("e"), true, vec![(Record::from("z"), 0.8)]),
        ],
        0.8,
        None,
    );
    assert!((d.duplicate_ratio() - 0.4).abs() < 1e-6);
    assert!((d.exact_duplicate_ratio() - 0.2).abs() < 1e-6);
    assert_eq!(d.get_least_similar_from_duplicates(1)[0].2, 0.8);

    d.rethreshold(0.85).unwrap();
    assert_eq!(d.filtered.len(), 1);
    assert_eq!(d.selected.len(), 4);
    assert!(d.rethreshold(0.80).is_err());
}

#[test]
fn selected_with_duplicates_is_cached_and_deduped() {
    let selected = rec(&[("id", 0.into()), ("text", "hello".into())]);
    let filtered = rec(&[("id", 1.into()), ("text", "hello".into())]);
    let selected_record = selected.clone();
    let filtered_record = filtered.clone();

    let mut d = DeduplicationResult::new(
        vec![selected],
        vec![
            DuplicateRecord::new(
                filtered.clone(),
                false,
                vec![(selected_record.clone(), 0.95)],
            ),
            DuplicateRecord::new(filtered, false, vec![(selected_record, 0.90)]),
        ],
        0.8,
        Some(vec!["text".to_string()]),
    );

    let items = d.selected_with_duplicates().to_vec();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].duplicates.len(), 1);
    assert_eq!(items[0].duplicates[0].0, filtered_record);
}

#[test]
fn from_records_rejects_dicts_without_columns_and_preserves_order() {
    let records = vec![rec(&[("text", "hello".into())])];
    assert!(SemHash::from_records(records, SemHashOptions::default().model(model())).is_err());

    let semhash = SemHash::from_records(
        text_records(&["zebra", "apple", "zebra", "banana", "apple", "cherry"]),
        SemHashOptions::default().model(model()),
    )
    .unwrap();
    let firsts: Vec<String> = semhash
        .index
        .items
        .iter()
        .map(|bucket| bucket[0].get("text").unwrap().as_string_lossy())
        .collect();
    assert_eq!(firsts, vec!["zebra", "apple", "banana", "cherry"]);
}

#[test]
fn single_dataset_deduplication_with_exact_and_semantic_duplicates() {
    let texts = text_records(&[
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",
        "It's risky to go alone!",
    ]);
    let semhash = SemHash::from_records(texts, SemHashOptions::default().model(model())).unwrap();
    let result = semhash.self_deduplicate(0.7).unwrap();
    assert_eq!(
        result.selected_texts().unwrap(),
        vec!["It's dangerous to go alone!"]
    );
    assert_eq!(result.filtered.len(), 2);
}

#[test]
fn cross_dataset_deduplication() {
    let train = text_records(&[
        "It's dangerous to go alone!",
        "It's a secret to everybody.",
        "Ganondorf has invaded Hyrule!",
    ]);
    let semhash = SemHash::from_records(train, SemHashOptions::default().model(model())).unwrap();
    let test = text_records(&[
        "It's dangerous to go alone!",
        "It's risky to go alone!",
        "Ganondorf has attacked Hyrule!",
    ]);
    let result = semhash.deduplicate(test, 0.7).unwrap();
    assert!(result.selected.is_empty());
    assert_eq!(result.filtered.len(), 3);
}

#[test]
fn multi_column_records_preserve_non_embedding_fields() {
    let records = vec![
        rec(&[
            ("id", 0.into()),
            ("text", "triforce".into()),
            ("metadata", "game1".into()),
        ]),
        rec(&[
            ("id", 1.into()),
            ("text", "master sword".into()),
            ("metadata", "game2".into()),
        ]),
        rec(&[
            ("id", 2.into()),
            ("text", "hylian shield".into()),
            ("metadata", "game3".into()),
        ]),
    ];
    let semhash = SemHash::from_records(
        records,
        SemHashOptions::default().columns(["text"]).model(model()),
    )
    .unwrap();
    let result = semhash.self_deduplicate(0.99).unwrap();
    assert_eq!(result.selected.len(), 3);
    for record in result.selected {
        let map = record.as_dict().unwrap();
        assert!(map.contains_key("id"));
        assert!(map.contains_key("metadata"));
    }
}

#[test]
fn outlier_filtering_and_ratios() {
    let train = text_records(&["apple", "banana", "cherry", "car", "bicycle"]);
    let semhash = SemHash::from_records(train, SemHashOptions::default().model(model())).unwrap();
    let test = text_records(&["apple", "banana", "kiwi", "motorcycle", "plane"]);
    let result = semhash.filter_outliers(test, 0.4).unwrap();
    assert_eq!(result.filtered.len(), 2);
    assert!((result.filter_ratio() - 0.4).abs() < 1e-6);
    assert!((result.selected_ratio() - 0.6).abs() < 1e-6);
    assert!(semhash.filter_outliers(Vec::new(), 1.5).is_err());
}

#[test]
fn representative_sampling_handles_auto_and_explicit_limits() {
    let train = text_records(&["apple", "banana", "cherry", "car", "bicycle"]);
    let semhash = SemHash::from_records(train, SemHashOptions::default().model(model())).unwrap();
    let result = semhash
        .self_find_representative(2, CandidateLimit::Auto, 0.5, Strategy::MMR)
        .unwrap();
    assert_eq!(result.selected.len(), 2);

    let result = semhash
        .self_find_representative(2, CandidateLimit::Value(3), 0.0, Strategy::MMR)
        .unwrap();
    assert_eq!(result.selected.len(), 2);
}

#[test]
fn from_embeddings_keeps_first_occurrence_embedding() {
    let records = text_records(&["apple", "banana", "apple", "cherry"]);
    let embeddings = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let semhash =
        SemHash::from_embeddings(embeddings, records, model(), SemHashOptions::default()).unwrap();
    assert_eq!(semhash.index.vectors, vec![vec![0.0], vec![1.0], vec![3.0]]);
}

#[test]
fn utility_functions_match_python_semantics() {
    assert_eq!(coerce_value(&42.into()), Value::String("42".into()));
    assert_eq!(coerce_value(&true.into()), Value::String("True".into()));
    assert_eq!(compute_candidate_limit_default(1000, 10), 100);
    assert_eq!(compute_candidate_limit_default(50, 10), 50);
    assert_eq!(compute_candidate_limit(20_000, 10, 0.1, 100, 1000), 1000);

    let prepared =
        semhash_rs::records::prepare_records(&text_records(&["hello", "world"]), None).unwrap();
    assert!(prepared.2);
    assert_eq!(prepared.1, vec!["text".to_string()]);
}
