"""Task 2 — Stage 1: supervised fine-tuning (PDF §2.2).

L_SFT(θ) = − (1 / #summary tokens) Σ_{t ∈ summary} log πθ(x_t | x_<t)
Prompt tokens and padding contribute nothing.
"""
import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

from common import (CKPT, RESULTS, SMOKE, DEVICE, autocast, make_scaler, optimizer_step, seed_all,
                    torch_generator, smoke, get_splits, load_tokenizer, load_model, save_policy,
                    token_logprobs, summary_mask, rollout, decode_responses, JsonlLogger, read_log,
                    write_summary, write_check, plot_panels)
from batching import collate_sft, collate_prompts

# ---- spec constants (§2.2) — do not change ----
LR = 5e-5
BATCH_SIZE = 16
EPOCHS = 1
GRAD_CLIP = 1.0
LOG_EVERY = 2 if SMOKE else 250
N_SAMPLE_POSTS = 5
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", BATCH_SIZE))   # OOM fallback; effective batch stays 16


@torch.inference_mode()
def validation_loss(model, loader):
    """Token-weighted mean NLL over every summary token of the validation set."""
    model.eval()
    total_nll, total_tokens = 0.0, 0.0
    for b in loader:
        ids, mask, plens = (b[k].to(DEVICE) for k in ("input_ids", "attention_mask", "prompt_lens"))
        m = summary_mask(mask, plens)
        with autocast():
            lp = token_logprobs(model, ids, mask, select=m)
        total_nll += float(-lp.sum())
        total_tokens += float(m.sum())
    model.train()
    return total_nll / total_tokens


def sft_backward(model, batch, scaler):
    """Backward pass for one optimiser step. The loss is normalised by the summary-token
    count of the whole batch, so splitting into micro-batches gives the exact same gradient."""
    ids, mask, plens = (batch[k].to(DEVICE) for k in ("input_ids", "attention_mask", "prompt_lens"))
    n_tokens = summary_mask(mask, plens).sum()
    step_loss = 0.0
    for rows in torch.arange(ids.size(0), device=DEVICE).split(MICRO_BATCH):
        m = summary_mask(mask[rows], plens[rows])
        with autocast():
            lp = token_logprobs(model, ids[rows], mask[rows], select=m)
        loss = -lp.sum() / n_tokens
        scaler.scale(loss).backward()
        step_loss += loss.item()
    return step_loss


@torch.no_grad()
def sample_summaries(model, tok, posts):
    model.eval()
    b = collate_prompts(posts, tok.pad_token_id)
    out = rollout(model, tok, b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), sample=False)
    model.train()
    return decode_responses(tok, out["resp"], out["rmask"])


def write_samples(posts, before, after):
    lines = ["# SFT: greedy summaries before / after fine-tuning (5 evaluation posts)\n"]
    for i, (p, a, b) in enumerate(zip(posts, before, after), 1):
        lines += [f"## Post {i}\n", "```", p["prompt"], "```\n",
                  f"- **Human reference:** {p['summary'].strip()}",
                  f"- **Base model (before SFT):** {a}",
                  f"- **SFT model (after):** {b}\n"]
    with open(os.path.join(RESULTS, "sft_samples.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    seed_all()
    tok = load_tokenizer()
    train, val = get_splits(tok, "sft")
    eval_posts = get_splits(tok, "eval")
    train, val = smoke(train, 64), smoke(val, 16)
    sample_posts = eval_posts[:N_SAMPLE_POSTS]

    collate = partial(collate_sft, pad_id=tok.pad_token_id)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                              generator=torch_generator())
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = load_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = make_scaler()
    logger = JsonlLogger("sft")

    before = sample_summaries(model, tok, sample_posts)
    logger.log(step=0, train_loss=None, val_loss=validation_loss(model, val_loader))

    t0, step, window = time.time(), 0, []
    for _ in range(EPOCHS):
        for batch in train_loader:
            step += 1
            step_loss = sft_backward(model, batch, scaler)
            grad_norm = optimizer_step(optimizer, scaler, model.parameters(), GRAD_CLIP)
            window.append(step_loss)
            if step % LOG_EVERY == 0 or step == len(train_loader) * EPOCHS:
                logger.log(step=step, train_loss=sum(window) / len(window),
                           val_loss=validation_loss(model, val_loader), grad_norm=grad_norm)
                window = []
    train_minutes = (time.time() - t0) / 60

    after = sample_summaries(model, tok, sample_posts)
    write_samples(sample_posts, before, after)
    save_policy(model, tok, CKPT["sft"])

    rows = [r for r in read_log("sft") if r["train_loss"] is not None]
    write_summary("sft", steps=step, final_train_loss=rows[-1]["train_loss"],
                  final_val_loss=rows[-1]["val_loss"], train_minutes=round(train_minutes, 1))
    write_check(f"[sft] trained {step} steps; final val loss {rows[-1]['val_loss']:.4f}")
    plot_panels("sft", [("SFT loss (summary tokens only)", ["train_loss", "val_loss"])], "fig1_sft_loss.png")


if __name__ == "__main__":
    main()
