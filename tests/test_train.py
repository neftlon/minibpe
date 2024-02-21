import pytest, os
from minibpe import NativeBasicTokenizer, PyBasicTokenizer
from minibpe import tokenizer_with_presplit, GPT4_SPLIT_PATTERN

@pytest.mark.parametrize("tokenizer_factory", [NativeBasicTokenizer, PyBasicTokenizer])
@pytest.mark.parametrize("train_text,test_text,vocab_size", [
  ("abc", "aabb", 256 + 2),
  ("foo bar baz", "foo foo", 256 + 3),
])
def test_mini_train(tokenizer_factory, train_text, test_text, vocab_size):
  tok = tokenizer_factory.train(train_text, vocab_size)
  assert tok.decode(tok.encode(test_text)) == test_text

@pytest.mark.parametrize("tokenizer_factory", [NativeBasicTokenizer, PyBasicTokenizer])
@pytest.mark.parametrize("test_text", ["", "1", "foo bar", "abc"])
def test_taylorswift_train(tokenizer_factory, test_text):
  # load training file and train tokenizer
  dirname = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dirname, "taylorswift.txt")
  contents = open(path, encoding="utf-8").read()
  tok = tokenizer_factory.train(contents, 256 + 3)

  assert tok.decode(tok.encode(test_text)) == test_text

def test_category_splitting():
  tok = tokenizer_with_presplit(
    "foo'nt?? bar't!!! ?? ...",
    256 + 15,
    GPT4_SPLIT_PATTERN,
  )
  assert len(tok.encode("'nt")) == 1

if __name__ == "__main__":
  pytest.main([__file__])
