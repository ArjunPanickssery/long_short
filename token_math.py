"""
from transformers import GPT2Tokenizer
def tokenize_gpt2(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    return tokens
"""

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def tokenize_gpt(text):
    return [enc.decode([token_idx]) for token_idx in enc.encode(text)]


# Example usage:
input_text = "Hello, world! This is a test."
tokens = tokenize_gpt(input_text)
print(tokens)
