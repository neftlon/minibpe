import tiktoken
from . import Gpt4Tokenizer

# all of this is copied from https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py


def bpe(mergeable_ranks, token, max_rank):
  # helper function used in get_gpt4_merges() to reconstruct the merge forest
  parts = [bytes([b]) for b in token]
  while True:
    min_idx = None
    min_rank = None
    for i, pair in enumerate(zip(parts[:-1], parts[1:])):
      rank = mergeable_ranks.get(pair[0] + pair[1])
      if rank is not None and (min_rank is None or rank < min_rank):
        min_idx = i
        min_rank = rank
    if min_rank is None or (max_rank is not None and min_rank >= max_rank):
      break
    assert min_idx is not None
    parts = (
      parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    )
  return parts


def recover_merges(mergeable_ranks):
  # the `merges` are already the byte sequences in their merged state.
  # so we have to recover the original pairings. We can do this by doing
  # a small BPE training run on all the tokens, in their order.
  # also see https://github.com/openai/tiktoken/issues/60
  # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
  merges = {}
  for token, rank in mergeable_ranks.items():
    if len(token) == 1:
      continue  # skip raw bytes
    pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
    assert len(pair) == 2
    # recover the integer ranks of the pair
    ix0 = mergeable_ranks[pair[0]]
    ix1 = mergeable_ranks[pair[1]]
    merges[(ix0, ix1)] = rank

  return merges


def create_gpt4_tokenizer(tiktoken_base_encoding="cl100k_base"):
  enc = tiktoken.get_encoding(tiktoken_base_encoding)
  mergeable_ranks = enc._mergeable_ranks
  merges = recover_merges(mergeable_ranks)
  byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
  return Gpt4Tokenizer.from_merges_and_shuffles(merges, byte_shuffle)
