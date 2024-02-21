use std::collections::HashMap;

use crate::{
    utils::{get_stats, get_top_pair, merge},
    BasicTokenizer, TrainError,
};
use fancy_regex::Regex;

fn get_splits<'t>(text: &'t str, regex: &str) -> impl IntoIterator<Item = &'t str> {
    let re = Regex::new(regex).expect("should be able to compile regex");
    // TODO: find out why the collect is necessary here
    re.find_iter(text)
        .map(|m| m.unwrap().as_str())
        .collect::<Vec<_>>()
}

fn get_stats_from_splits(splits: &Vec<Vec<u32>>) -> HashMap<(u32, u32), usize> {
    let mut stats = HashMap::new();
    for ids in splits {
        let local_stats = get_stats(&ids);
        stats.extend(local_stats.iter());
    }
    stats
}

fn merge_in_splits(splits: &Vec<Vec<u32>>, pair: (u32, u32), idx: u32) -> Vec<Vec<u32>> {
    let mut res = vec![];
    for split in splits {
        let split = merge(split, pair, idx);
        res.push(split);
    }
    res
}

/// Creates a [`BasicTokenizer`] but splits the text before generating merges.
pub fn tokenizer_with_presplit(
    text: &str,
    vocab_size: u32,
    regex: &str,
) -> Result<BasicTokenizer, TrainError> {
    // Create splits and convert them to identifiers
    let splits = get_splits(text, regex);
    let splits = splits.into_iter().map(|s| s.as_bytes());
    let mut splits: Vec<_> = splits
        .map(|bytes| bytes.into_iter().map(|c| *c as u32).collect::<Vec<_>>())
        .collect();

    // run BPE
    assert!(
        vocab_size >= 256,
        "vocabulary must include at least all bytes (>= 256)"
    );
    let num_merges = vocab_size - 256;
    let mut merges = HashMap::new();
    for idx in 0..num_merges {
        let stats = get_stats_from_splits(&splits);
        if let Some((pair, _)) = get_top_pair(&stats) {
            let idx = 256 + idx; // start new tokens at index 256
            splits = merge_in_splits(&splits, pair, idx); // replace pair
            merges.insert(pair, idx); // store merge
        } else {
            return Err(TrainError::NotEnoughPairs);
        }
    }
    Ok(BasicTokenizer::from_merges(&merges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fancy_regex::Error;

    // GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

    #[test]
    fn it_works() -> Result<(), TrainError> {
        let tok = tokenizer_with_presplit("foo bar'nt! fo.", 256 + 5, GPT4_SPLIT_PATTERN)?;
        assert_eq!(tok.encode("fo").len(), 1);
        Ok(())
    }

    #[test]
    fn gpt4_split_pattern() -> Result<(), Error> {
        let re = Regex::new(GPT4_SPLIT_PATTERN)?;
        assert_eq!(
            re.find_iter("foo bar'nt!")
                .map(|m| m.unwrap().as_str())
                .collect::<Vec<_>>(),
            vec!["foo", " bar", "'nt", "!"],
        );
        Ok(())
    }
}
