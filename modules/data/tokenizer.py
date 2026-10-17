from __future__ import annotations

from transformers import AutoTokenizer
from torch import Tensor

VOCAB_SIZE: int = 50257
EOS_TOKEN_ID: int = 50256
MODEL_NAME: str = "gpt2"


def get_tokenizer():
    """Return the GPT-2 BPE tokenizer.

    Outsourced on purpose: GPT-2's vocabulary is 50,257 learned merge ranks.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(text: str, tokenizer) -> list[int]:
    """Text -> token ids"""
    return tokenizer(text)["input_ids"]


def decode(ids: Tensor | list[int], tokenizer) -> str:
    """Token ids -> text"""
    return tokenizer.decode(ids)


def vocab_size(tokenizer) -> int:
    return len(tokenizer)
