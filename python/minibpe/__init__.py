from minibpe._native import NativeBasicTokenizer
from minibpe.basic import PyBasicTokenizer
from minibpe._native import py_tokenizer_with_presplit as tokenizer_with_presplit
BasicTokenizer = NativeBasicTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
