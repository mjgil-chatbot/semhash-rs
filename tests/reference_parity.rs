use semhash_rs::*;
use std::collections::BTreeMap;
use std::sync::Arc;

const FRUITS: &[&str] = &[
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
];

const VEHICLES: &[&str] = &["car", "bicycle", "motorcycle", "plane"];

#[derive(Debug)]
struct ToyEncoder;

impl Encoder for ToyEncoder {
    fn encode(&self, inputs: &[Value]) -> Result<Embeddings> {
        Ok(inputs.iter().map(embed_value).collect())
    }
}

fn embed_value(value: &Value) -> Vec<f32> {
    let text = value.as_string_lossy().to_lowercase();
    if FRUITS.contains(&text.as_str()) {
        vec![1.0, 0.0, 0.0]
    } else if VEHICLES.contains(&text.as_str()) {
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
        vec![0.0, 0.0, 1.0]
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

fn dict_record_text(record: &Record) -> String {
    record
        .as_dict()
        .and_then(|record| record.get("text"))
        .map(Value::as_string_lossy)
        .expect("expected text field")
}

#[test]
fn representative_and_outlier_results_match_reference_oracles() {
    let train = text_records(&[
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
        "car",
        "bicycle",
    ]);
    let test = text_records(&[
        "apple",
        "banana",
        "kiwi",
        "fig",
        "apricot",
        "grapefruit",
        "pomegranate",
        "motorcycle",
        "plane",
    ]);
    let semhash = SemHash::from_records(train, SemHashOptions::default().model(model())).unwrap();

    let self_representative = semhash
        .self_find_representative(3, CandidateLimit::Value(5), 0.5, Strategy::MMR)
        .unwrap();
    let mut self_selected: Vec<String> = self_representative
        .selected
        .iter()
        .map(dict_record_text)
        .collect();
    self_selected.sort();
    assert_eq!(self_selected, vec!["apple", "banana", "cherry"]);

    let representative = semhash
        .find_representative(
            test.clone(),
            3,
            CandidateLimit::Value(5),
            0.5,
            Strategy::MMR,
        )
        .unwrap();
    let mut selected: Vec<String> = representative
        .selected
        .iter()
        .map(dict_record_text)
        .collect();
    selected.sort();
    assert_eq!(selected, vec!["apple", "banana", "kiwi"]);

    let self_outliers = semhash.self_filter_outliers(0.1).unwrap();
    let mut self_filtered: Vec<String> = self_outliers
        .filtered
        .iter()
        .map(dict_record_text)
        .collect();
    self_filtered.sort();
    assert_eq!(self_filtered, vec!["bicycle", "car"]);

    let outliers = semhash.filter_outliers(test, 0.2).unwrap();
    let mut filtered: Vec<String> = outliers.filtered.iter().map(dict_record_text).collect();
    filtered.sort();
    assert_eq!(filtered, vec!["motorcycle", "plane"]);
}

#[test]
fn deduplicate_edge_cases_match_reference_errors() {
    let semhash = SemHash::from_records(
        text_records(&["1", "2", "3"]),
        SemHashOptions::default().model(model()),
    )
    .unwrap();

    let result = semhash
        .deduplicate(
            vec![rec(&[("text", 1.into())]), rec(&[("text", 4.into())])],
            0.95,
        )
        .unwrap();
    assert_eq!(result.selected.len() + result.filtered.len(), 2);

    let err = semhash
        .deduplicate(
            vec![
                rec(&[("text", "cherry".into())]),
                rec(&[("text", Value::Null)]),
            ],
            0.95,
        )
        .unwrap_err();
    assert!(err.to_string().contains("has None value"));

    let err = semhash.deduplicate(Vec::new(), 0.95).unwrap_err();
    assert_eq!(err.to_string(), "records must not be empty");

    let semhash_dict = SemHash::from_records(
        vec![rec(&[("col", "a".into())]), rec(&[("col", "b".into())])],
        SemHashOptions::default().columns(["col"]).model(model()),
    )
    .unwrap();

    let err = semhash_dict
        .deduplicate(text_records(&["x", "y"]), 0.95)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Records were not originally strings, but you passed strings."
    );

    let err = semhash
        .deduplicate(vec![Record::from("a"), rec(&[("text", "b".into())])], 0.95)
        .unwrap_err();
    assert_eq!(err.to_string(), "All records must be all strings.");

    let err = semhash_dict
        .deduplicate(vec![rec(&[("col", "a".into())]), Record::from("b")], 0.95)
        .unwrap_err();
    assert_eq!(err.to_string(), "All records must be all dictionaries.");
}

#[test]
fn selected_with_duplicates_cache_invalidation_matches_reference() {
    let mut result = DeduplicationResult::new(
        vec![Record::from("original")],
        vec![
            DuplicateRecord::new(
                Record::from("duplicate_1"),
                false,
                vec![(Record::from("original"), 0.9)],
            ),
            DuplicateRecord::new(
                Record::from("duplicate_2"),
                false,
                vec![(Record::from("original"), 0.8)],
            ),
            DuplicateRecord::new(
                Record::from("duplicate_3"),
                false,
                vec![(Record::from("original"), 0.7)],
            ),
        ],
        0.7,
        None,
    );

    let before = result.selected_with_duplicates().to_vec();
    assert_eq!(before[0].duplicates.len(), 3);

    result.rethreshold(0.85).unwrap();

    let after = result.selected_with_duplicates().to_vec();
    assert_eq!(after[0].duplicates.len(), 1);
    assert_eq!(after[0].duplicates[0].0, Record::from("duplicate_1"));
    assert_ne!(before, after);
}

#[test]
fn empty_filter_result_matches_reference_ratios() {
    let result = FilterResult::new(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    assert_eq!(result.filter_ratio(), 0.0);
    assert_eq!(result.selected_ratio(), 1.0);
}
