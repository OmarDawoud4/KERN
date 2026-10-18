import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.data.dataset import TextDataset
from modules.kern1 import KERN1


def get_device()-> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, opt, step: int, best_val: float, scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, model, opt, scaler=None) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["best_val"]


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    """Linear warmup for `warmup` steps, then cosine decay to base_lr * min_frac."""
    floor = base_lr * min_frac
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / (total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return floor + (base_lr - floor) * coeff


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None, amp_dtype=None) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with ctx:
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main()->None : 
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("runs/kern"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-eval-batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--amp", action="store_true", help="autocast fp16/bf16 for tensor cores")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = get_device()
    train_tokens = torch.load(args.data / "train.pt")
    val_tokens = torch.load(args.data / "val.pt")


    train_ds = TextDataset(train_tokens, args.block_size)
    val_ds = TextDataset(val_tokens, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) 

    model = KERN1(vocab_size=50257).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # fp16 needs loss scaling; bf16 does not. Turing (T4) has no bf16, so fp16 there.
    # Pre-Turing (P100/K80) has weak fp16, so leave amp off there.
    amp_dtype = None
    if args.amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

    ckpt_path = args.out / "checkpoint.pt"
    start_step, best_val = 0, float("inf")
    if args.resume and ckpt_path.exists():
        start_step, best_val = load_checkpoint(ckpt_path, model, opt, scaler)
        print(f"resumed from step {start_step}, best val {best_val:.4f}")

    print(f"device {device} | amp {amp_dtype or 'off'} | params {sum(p.numel() for p in model.parameters()):,}")
    print(f"train examples {len(train_ds):,} | val examples {len(val_ds):,}")



    step = start_step
    running =0.0
    seen = 0
    t0 = time.time()

    while step< args.steps:
        for x , y in train_loader:
            if step >= args.steps:
                break

            x,y = x.to(device), y.to(device)

            cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
            for group in opt.param_groups:
                group["lr"] = cur_lr

            amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
            with amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item() 
            seen += 1
            step += 1

            if step % args.eval_every == 0 or step ==args.steps:
                val= evaluate(model , val_loader , device , args.max_eval_batches , amp_dtype )
                train_avg = running /max(seen , 1)
                best_val = min(best_val, val)

                print(
                    f"step {step:>6} | train {train_avg:.4f} | val {val:.4f} "
                    f"| lr {cur_lr:.2e} | {time.time() - t0:6.1f}s"
                )
                running, seen = 0.0, 0

            if step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, opt, step, best_val, scaler)

    save_checkpoint(ckpt_path, model, opt, step, best_val, scaler)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()


