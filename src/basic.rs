use std::collections::HashMap;

fn merge(tokens: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut res = vec![];
    let mut i = 0;
    while i < tokens.len() {
        if i < (tokens.len() - 1) && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            res.push(idx);
            i += 2;
        } else {
            res.push(tokens[i]);
            i += 1;
        }
    }
    res
}

fn get_stats(ids: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut res = HashMap::new();
    for pair in ids.iter().copied().zip(ids.iter().skip(1).copied()) {
        if let Some(count) = res.get_mut(&pair) {
            *count += 1;
        } else {
            res.insert(pair, 1);
        }
    }
    res
}

fn get_top_pair(stats: &HashMap<(u32, u32), usize>) -> Option<((u32, u32), usize)> {
    // TODO: is there a way in rust to immediately/automatically copy trivially
    // copy-able elements to get rid of `.map(|(p, c)| (*p, *c))`?
    stats
        .iter()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(p, c)| (*p, *c))
}

fn create_vocab_from_merges<I: Clone + IntoIterator<Item = ((u32, u32), u32)>>(
    merges: &I,
) -> HashMap<u32, Vec<u8>> {
    // create trivial vocab from bytes
    let mut vocab = HashMap::new();
    for idx in 0..256 {
        vocab.insert(idx, vec![idx as u8]);
    }

    // insert more complex items from merges
    let mut sorted_merges: Vec<_> = merges.clone().into_iter().collect();
    sorted_merges.sort_by(|(_, idx1), (_, idx2)| idx1.cmp(idx2));
    for ((p1, p2), idx) in sorted_merges {
        // Safety: Keys should either be atomic (0..256) or be inserted by
        // previous merges (ensured by the ordering).
        let p1 = vocab.get(&p1).unwrap();
        let p2 = vocab.get(&p2).unwrap();
        // concatenate two representations to generate new one
        let new = p1
            .into_iter()
            .copied()
            .chain(p2.into_iter().copied())
            .collect();
        vocab.insert(idx, new);
    }

    vocab
}

#[derive(Clone, Debug, Default, PartialEq)]
struct BasicTokenizer {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
enum TrainError {
    NotEnoughPairs,
}

impl BasicTokenizer {
    fn from_merges<I: Clone + IntoIterator<Item = ((u32, u32), u32)>>(merges: &I) -> Self {
        let merges = HashMap::from_iter(merges.clone().into_iter());
        let vocab = create_vocab_from_merges(&merges.clone());
        Self { merges, vocab }
    }

    fn train(text: &str, vocab_size: u32) -> Result<Self, TrainError> {
        assert!(
            vocab_size >= 256,
            "vocabulary must include at least all bytes (>= 256)"
        );
        let num_merges = vocab_size - 256;
        let mut tokens: Vec<_> = text.as_bytes().into_iter().map(|c| *c as u32).collect();
        let mut merges = HashMap::new();
        for idx in 0..num_merges {
            let stats = get_stats(&tokens);
            if let Some((pair, _)) = get_top_pair(&stats) {
                let idx = 256 + idx; // start new tokens at index 256
                tokens = merge(&tokens, pair, idx); // replace pair
                merges.insert(pair, idx); // store merge
            } else {
                return Err(TrainError::NotEnoughPairs);
            }
        }
        Ok(Self::from_merges(&merges))
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Safety: Conversion from char to u32 should always be valid.
        let mut tokens: Vec<_> = text.as_bytes().into_iter().map(|c| *c as u32).collect();
        while let Some(pair) = {
            let stats = get_stats(&tokens);
            // get pair with minimum merge index (indicated by lowest generated
            // identifier).
            // Safety: The unwrap is ok since we filter the list to allow only pairs
            // that can actually be merged.
            stats
                .iter()
                .filter(|(&pair, _)| self.merges.contains_key(&pair))
                .min_by(|(pair1, _), (pair2, _)| {
                    self.merges
                        .get(pair1)
                        .unwrap()
                        .cmp(self.merges.get(pair2).unwrap())
                })
                .map(|(p, _)| *p)
        } {
            let idx = self
                .merges
                .get(&pair)
                .expect("we should find and index to the merge");
            tokens = merge(&tokens, pair, *idx);
        }
        tokens
    }

    fn decode(&self, tokens: impl IntoIterator<Item = u32>) -> String {
        let mut res = Vec::default();
        for token in tokens.into_iter() {
            let bytes = self
                .vocab
                .get(&token)
                .expect("vocab should contain a mapping for every token");
            res.extend_from_slice(&bytes);
        }
        String::from_utf8_lossy(&res).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<(), TrainError> {
        let tok = BasicTokenizer::train("abc", 256 + 2)?;
        let tokens = tok.encode("abc");
        assert_eq!(tokens, vec![257]);
        assert_eq!(tok.decode(tokens), "abc");
        Ok(())
    }

    #[test]
    fn no_merge_error() {
        assert_eq!(
            BasicTokenizer::train("", 256 + 1),
            Err(TrainError::NotEnoughPairs)
        );
        assert_eq!(
            BasicTokenizer::train("a", 256 + 1),
            Err(TrainError::NotEnoughPairs)
        );
        assert_eq!(
            BasicTokenizer::train("ab", 256 + 2),
            Err(TrainError::NotEnoughPairs)
        );
    }

    #[test]
    fn vocab_creation() {
        let merges = HashMap::from([((1, 2), 256), ((256, 3), 257)]);

        let mut vocab = HashMap::new();
        // insert "trivial" items
        for idx in 0..256 {
            vocab.insert(idx, vec![idx as u8]);
        }
        // insert rules induced by merges
        vocab.insert(256, vec![1, 2]);
        vocab.insert(257, vec![1, 2, 3]);
        assert_eq!(create_vocab_from_merges(&merges), vocab);
    }

    #[test]
    fn preserve_characters() {
        // create a tokenizer without any merges
        let notok = BasicTokenizer::default();
        let text = "foo bar baz åð";
        let ids1 = notok.encode(&text).into_iter().collect::<Vec<_>>();
        let ids2 = text
            .as_bytes()
            .into_iter()
            .map(|c| *c as u32)
            .collect::<Vec<_>>();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn encode_loop() {
        let ids = [1, 1, 2, 3];
        let merges = HashMap::from([((1, 2), 98), ((98, 3), 99)]);

        let mut tokens = Vec::from(ids);
        while let Some(pair) = {
            let stats = get_stats(&tokens);
            // get pair with minimum merge index (indicated by lowest generated
            // identifier).
            // Safety: The unwrap is ok since we filter the list to allow only pairs
            // that can actually be merged.
            stats
                .iter()
                .filter(|(&pair, _)| merges.contains_key(&pair))
                .min_by(|(pair1, _), (pair2, _)| {
                    merges.get(pair1).unwrap().cmp(merges.get(pair2).unwrap())
                })
                .map(|(p, _)| *p)
        } {
            let idx = merges
                .get(&pair)
                .expect("we should find and index to the merge");
            tokens = merge(&tokens, pair, *idx);
        }

        assert_eq!(tokens, vec![1, 99]);
    }

    #[test]
    fn find_merge() {
        let ids = [1, 1, 2, 3];
        let merges = HashMap::from([((1, 2), 98), ((98, 3), 99)]);

        let stats = get_stats(&ids);
        assert_eq!(
            stats,
            HashMap::from([((1, 1), 1), ((1, 2), 1), ((2, 3), 1)])
        );

        // get pair with minimum merge index (indicated by lowest generated
        // identifier).
        // Safety: The unwrap is ok since we filter the list to allow only pairs
        // that can actually be merged.
        let pair = stats
            .iter()
            .filter(|(&pair, _)| merges.contains_key(&pair))
            .min_by(|(pair1, _), (pair2, _)| {
                merges.get(pair1).unwrap().cmp(merges.get(pair2).unwrap())
            })
            .map(|(p, _)| *p);
        assert!(
            pair.is_some(),
            "there should be at least one merge-able pair"
        );
        let pair = pair.unwrap();

        let idx = merges.get(&pair).expect("we should find 98");
        assert_eq!(*idx, 98);
    }

    #[test]
    fn replace_top_pair() {
        let ids = [1, 2, 3, 1, 2, 1, 1, 2];

        // extract the top-pair
        let stats = get_stats(&ids);
        let top_pair = get_top_pair(&stats);
        let top_pair = top_pair.expect("there should be a top-pair in sequence of length > 1");
        assert_eq!(top_pair, ((1, 2), 3));

        // replace top-pair with a new token
        assert_eq!(merge(&ids, top_pair.0, 4), vec![4, 3, 4, 1, 4]);
    }

    #[test]
    fn merges() {
        assert_eq!(merge(&[], (1, 2), 3), vec![]);
        assert_eq!(merge(&[1, 2, 2, 3], (1, 2), 4), vec![4, 2, 3]);
        assert_eq!(merge(&[1, 2, 2, 2, 3], (2, 2), 4), vec![1, 4, 2, 3])
    }

    #[test]
    fn counts() {
        assert_eq!(get_stats(&[]), HashMap::default());
        assert_eq!(
            get_stats(&[1, 2, 2, 2, 3]),
            HashMap::from([((1, 2), 1), ((2, 2), 2), ((2, 3), 1),]),
        );
    }
}
