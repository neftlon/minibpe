import pytest, os
from minibpe import BasicTokenizer

@pytest.mark.parametrize("test_text", ["", "1", "foo bar"])
def test_mini_train(test_text):
  # load training file and train tokenizer
  dirname = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(dirname, "taylorswift.txt")
  contents = open(path, encoding="utf-8").read()
  tok = BasicTokenizer.train(contents, 256 + 30)

  assert tok.decode(tok.encode(test_text)) == test_text

if __name__ == "__main__":
  pytest.main([__file__])
