import os

import tiktoken
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


gpt_encoding = tiktoken.get_encoding("cl100k_base")
llama2_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", use_auth_token=HF_TOKEN
)
llama3_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=HF_TOKEN
)


def clean_token_text(token_text):
    return token_text.replace("Ġ", " ").replace("▁", " ").strip()


def tokenize_gpt(text):
    return [
        clean_token_text(gpt_encoding.decode([token_idx]))
        for token_idx in gpt_encoding.encode(text)
    ]


def tokenize_llama2(text):
    return [
        clean_token_text(t)
        for t in llama2_tokenizer.tokenize(text)
        if clean_token_text(t)
    ]


def tokenize_llama3(text):
    return [
        clean_token_text(t)
        for t in llama3_tokenizer.tokenize(text)
        if clean_token_text(t)
    ]
