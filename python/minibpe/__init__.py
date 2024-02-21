from minibpe._native import NativeBasicTokenizer, NativeGpt4Tokenizer
from minibpe._native import py_tokenizer_with_presplit as tokenizer_with_presplit
from minibpe.basic import PyBasicTokenizer

BasicTokenizer = NativeBasicTokenizer
Gpt4Tokenizer = NativeGpt4Tokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
