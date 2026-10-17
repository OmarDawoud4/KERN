"""Build a tokenized TinyStories tensor for KERN-GPT.

Streams stories, tokenizes with GPT-2 BPE, concatenates into one flat
token stream separated by <|endoftext|>, and saves to disk as uint16.

Tokens are stored as uint16 because the largest GPT-2 id is 50256 < 65536.
That halves disk and RAM versus int32.

"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from datasets import load_dataset

from modules.data.tokenizer import EOS_TOKEN_ID, encode, get_tokenizer

DATASET = "roneneldan/TinyStories"
DEFAULT_OUT = Path("data")


def collect_stories(n_stories: int) -> list[str]:
    """Stream the first n_stories from the dataset without downloading 1.9 GB."""
    ds = load_dataset(DATASET, split="train", streaming=True)
    texts = (row["text"] for row in ds)
    return list(itertools.islice(texts, n_stories))


def build_tokens(stories: list[str], tokenizer) -> torch.Tensor:
    """Tokenize all stories into one flat uint16 stream with EOS separators."""
    ids: list[int] = []
    for story in stories:
        ids.extend(encode(story, tokenizer))
        ids.append(EOS_TOKEN_ID)
    return torch.tensor(ids, dtype=torch.uint16)


def split(train: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split at a document boundary, not mid-story."""
    n_val = int(len(train) * val_fraction)
    if n_val == 0:
        return train, train[:0]
    # Walk back to the nearest EOS so the validation set starts on a clean story.
    start = len(train) - n_val
    while start < len(train) and int(train[start]) != EOS_TOKEN_ID:
        start += 1
    # clone() is required: a slice is a view sharing the original storage, and
    # torch.save would then write the whole underlying tensor to disk.
    return train[:start].clone(), train[start:].clone()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-stories", type=int, default=5000)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tokenizer = get_tokenizer()
    print(f"tokenizer : gpt2, vocab {len(tokenizer)}")

    stories = collect_stories(args.n_stories)
    chars = sum(len(s) for s in stories)
    print(f"fetched   : {len(stories)} stories, {chars:,} chars")

    tokens = build_tokens(stories, tokenizer)
    print(f"tokenized : {len(tokens):,} tokens "
          f"({chars / len(tokens):.2f} chars/token)")

    train, val = split(tokens, args.val_fraction)
    print(f"split     : train {len(train):,} | val {len(val):,}")

    torch.save(train, args.out / "train.pt")
    torch.save(val, args.out / "val.pt")
    print(f"saved     : {args.out}/train.pt, {args.out}/val.pt")
    print(f"elapsed   : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
