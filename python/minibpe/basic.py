import typing

def merge(tokens, pair, id):
  i, new_tokens = 0, []
  while i < len(tokens):
    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
      new_tokens.append(id)
      i += 2
    else:
      new_tokens.append(tokens[i])
      i += 1
  return new_tokens

def get_stats(ids):
  stats = {}
  for pair in zip(ids, ids[1:]):
    stats[pair] = stats.get(pair, 0) + 1
  return stats

class PyBasicTokenizer(typing.NamedTuple):
  merges: typing.Mapping[tuple[int, int], int]
  vocab: typing.Mapping[int, bytes]

  @classmethod
  def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "PyBasicTokenizer":
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # generate merges
    tokens = list(text.encode("utf-8"))
    merges = {}
    for i in range(num_merges):
      stats = get_stats(tokens)
      assert stats, "NotEnoughPairs"
      top_pair = max(stats, key=stats.get)
      new_token = 256 + i
      tokens = merge(tokens, top_pair, new_token)
      if verbose:
        print("merging %4d and %4d (freq=%d) into new token %4d"
              % (*top_pair, stats[top_pair], new_token))
      merges[top_pair] = new_token

    # generate vocab
    vocab = {idx: bytes([idx]) for idx in range(256)} # generate trivial vocab
    for (p0,p1), idx in merges.items(): # this loop needs to be in the same order as the inserts
      vocab[idx] = vocab[p0] + vocab[p1]
    
    return cls(merges=merges, vocab=vocab)

  def encode(self, text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break # nothing else to merge
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx)
    return tokens

  def decode(self, tokens):
    tokens = b"".join(self.vocab[t] for t in tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text
