use std::collections::HashMap;

pub(crate) fn merge(tokens: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
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

pub(crate) fn get_stats(ids: &[u32]) -> HashMap<(u32, u32), usize> {
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

pub(crate) fn get_top_pair(stats: &HashMap<(u32, u32), usize>) -> Option<((u32, u32), usize)> {
    // TODO: is there a way in rust to immediately/automatically copy trivially
    // copy-able elements to get rid of `.map(|(p, c)| (*p, *c))`?
    stats
        .iter()
        .max_by(|a, b| a.1.cmp(b.1))
        .map(|(p, c)| (*p, *c))
}
