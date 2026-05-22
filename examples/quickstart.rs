use semhash_rs::{Record, SemHash, SemHashOptions};

fn main() -> semhash_rs::Result<()> {
    let records = vec![
        Record::from("It's dangerous to go alone!"),
        Record::from("It's dangerous to go alone!"),
        Record::from("It's risky to go alone!"),
    ];

    let semhash = SemHash::from_records(records, SemHashOptions::default())?;
    let result = semhash.self_deduplicate(0.90)?;

    println!("selected: {:?}", result.selected);
    println!("filtered: {:?}", result.filtered);
    println!("duplicate ratio: {}", result.duplicate_ratio());
    Ok(())
}
