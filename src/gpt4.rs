use std::collections::HashMap;

use crate::BasicTokenizer;

fn invert_mapping(mapping: &HashMap<u8, u8>) -> HashMap<u8, u8> {
    let mut res = HashMap::new();
    for (&k, &v) in mapping {
        res.insert(v, k);
    }
    res
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gpt4Tokenizer {
    /// Wrapped tokenizer.
    tok: BasicTokenizer,
    byte_shuffle: HashMap<u8, u8>,
    inverse_byte_shuffle: HashMap<u8, u8>,
}

impl Gpt4Tokenizer {
    pub fn from_merges_and_shuffles<I: Clone + IntoIterator<Item = ((u32, u32), u32)>>(
        merges: &I,
        byte_shuffle: &HashMap<u8, u8>,
    ) -> Self {
        Self {
            tok: BasicTokenizer::from_merges(merges),
            byte_shuffle: byte_shuffle.clone(),
            inverse_byte_shuffle: invert_mapping(&byte_shuffle),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut bytes: Vec<_> = text.as_bytes().into_iter().copied().collect();
        // replace bytes by mapped characters
        for byte in bytes.iter_mut() {
            *byte = *self
                .byte_shuffle
                .get(byte)
                .expect("mapping does not contain entry for token");
        }
        self.tok.encode_from_bytes(&bytes)
    }

    pub fn decode(&self, tokens: impl IntoIterator<Item = u32>) -> String {
        let mut bytes = self.tok.decode_into_bytes(tokens);
        // replace bytes by mapped characters
        for byte in bytes.iter_mut() {
            *byte = *self
                .inverse_byte_shuffle
                .get(byte)
                .expect("inverse mapping does not contain entry for byte");
        }
        String::from_utf8_lossy(&bytes).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_map() -> HashMap<u8, u8> {
        let mut res = HashMap::new();
        for i in 0..255 {
            res.insert(i, i);
        }
        res
    }

    #[test]
    fn shuffle_encode_decode() {
        fn swap(map: &mut HashMap<u8, u8>, a: u8, b: u8) {
            *map.get_mut(&a).unwrap() = b;
            *map.get_mut(&b).unwrap() = a;
        }

        let mut byte_shuffle = identity_map();
        // switch code points for a,b and c,d
        swap(&mut byte_shuffle, 97, 98);
        swap(&mut byte_shuffle, 99, 100);

        // create a tokenizer that does not do anything but flip two bytes
        let dummy_tok = Gpt4Tokenizer {
            tok: BasicTokenizer::from_merges(&HashMap::from([])),
            byte_shuffle: byte_shuffle.clone(),
            inverse_byte_shuffle: invert_mapping(&byte_shuffle),
        };

        assert_eq!(dummy_tok.decode([98, 97, 100, 99, 101]), "abcde");
        assert_eq!(dummy_tok.encode("abcde"), vec![98, 97, 100, 99, 101]);
    }

    #[test]
    fn create_inverse_mapping() {
        assert_eq!(
            invert_mapping(&HashMap::from([(1, 2), (3, 4), (5, 6)])),
            HashMap::from([(2, 1), (4, 3), (6, 5)]),
        );
    }
}
