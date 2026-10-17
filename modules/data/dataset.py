"""Next-token dataset for KERN-GPT.

Turns the flat token stream from prepare.py into (input, target) pairs.

A window of `block_size` tokens at position i pairs with the same window
shifted one step, so the model learns to predict the next token.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size].long()
        y = self.data[idx + 1 : idx + 1 + self.block_size].long()
        return x, y
