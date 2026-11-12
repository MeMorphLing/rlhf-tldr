"""Shared helpers for every stage — the single source of truth for alignment,
masking, generation, logging and plotting.

Index convention used everywhere:
    token_logprobs(...)[:, j] = log π(x_{j+1} | x_{≤j})          shape (B, L-1)
so the log-prob of the token at absolute position t lives in column t-1.

Run `python common.py` to execute Task 1 (data statistics + summary-mask check).
"""
import contextlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from config import (MODEL_NAME, SEED, MAX_SUMMARY_TOKENS,  # noqa: E402
                    N_SFT_TRAIN, N_SFT_VAL, N_PREF_TRAIN, N_PREF_VAL, N_EVAL_POSTS)
from models import load_tokenizer, load_causal_lm  # noqa: E402

# ----------------------------------------------------------------------------- paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = {k: os.path.join(ROOT, "checkpoints", k) for k in ("sft", "reward", "ppo", "dpo")}
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
FIGS = os.path.join(RESULTS, "figures")
CACHE = os.path.join(ROOT, ".cache")
for _d in (LOGS, FIGS, CACHE):
    os.makedirs(_d, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"          # tiny end-to-end run for debugging
USE_AMP = DEVICE == "cuda" and os.environ.get("RLHF_AMP", "1") == "1"


# ----------------------------------------------------------------------------- reproducibility
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_generator(seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def smoke(items, n):
    """Slice a split down to n items when SMOKE=1, otherwise return it unchanged."""
    return items[:n] if SMOKE else items


# ----------------------------------------------------------------------------- mixed precision
def autocast():
    """fp16 autocast on CUDA (T4 has no fast bf16); no-op on CPU or with RLHF_AMP=0."""
    if USE_AMP:
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler():
    return torch.amp.GradScaler("cuda", enabled=USE_AMP)


def optimizer_step(optimizer, scaler, params, max_norm=None):
    """Unscale → (clip) → step → update. max_norm=None means no clipping (the norm is
    still measured for logging). Returns the pre-clip gradient norm."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, float("inf") if max_norm is None else max_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


# ----------------------------------------------------------------------------- data
def _cache_path(name):
    return os.path.join(CACHE, f"{name}.json")


def get_splits(tok, which):
    """which ∈ {'sft', 'pref', 'eval'}. Identical output to the starter loaders,
    cached as JSON so later stages don't re-download and re-tokenise."""
    from data import load_sft_splits, load_preference_splits, load_eval_posts
    path = _cache_path(which)
    if os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
        return tuple(out) if which in ("sft", "pref") else out
    loader = {"sft": load_sft_splits, "pref": load_preference_splits, "eval": load_eval_posts}[which]
    out = loader(tok)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def build_prompt(subreddit, title, post):
    """Format a raw post exactly like the dataset prompts, then clean it with the starter."""
    from data import clean_post
    sub = subreddit.strip().removeprefix("r/")
    return clean_post(f"SUBREDDIT: r/{sub}\nTITLE: {title.strip()}\nPOST: {post.strip()}")


# ----------------------------------------------------------------------------- models
def load_model(path=MODEL_NAME, device=DEVICE):
    """Starter loader + fp32 master weights (transformers ≥5 loads the bf16 checkpoint as bf16)."""
    return load_causal_lm(path, device).float()


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def assert_frozen(name, model):
    n_trainable = sum(p.requires_grad for p in model.parameters())
    assert n_trainable == 0, f"{name}: {n_trainable} parameters still require grad"
    write_check(f"[frozen] {name}: all {sum(1 for _ in model.parameters())} parameter tensors "
                f"have requires_grad=False")


def save_policy(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- padding helpers
def position_ids_from_mask(attention_mask):
    """Positions that skip padding. Required for left-padded inputs so a plain
    forward pass sees the same positions model.generate() used."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def last_real_index(attention_mask):
    """Index of the last real token, for left or right padding. pad == eos in this
    tokenizer, so real tokens are found from the attention mask only."""
    L = attention_mask.size(1)
    return (attention_mask.long() * torch.arange(L, device=attention_mask.device)).argmax(1)


def pad_right(x, length, value):
    return torch.nn.functional.pad(x, (0, length - x.size(1)), value=value)


def stack_pairs(batch, pad_id):
    """collate_pref output → one right-padded (2B, L) batch: chosen rows first, then rejected."""
    L = max(batch["chosen_ids"].size(1), batch["rejected_ids"].size(1))
    ids = torch.cat([pad_right(batch["chosen_ids"], L, pad_id), pad_right(batch["rejected_ids"], L, pad_id)])
    mask = torch.cat([pad_right(batch["chosen_mask"], L, 0), pad_right(batch["rejected_mask"], L, 0)])
    plens = torch.cat([batch["prompt_lens"], batch["prompt_lens"]])
    return ids, mask, plens


# ----------------------------------------------------------------------------- log-probs & masks
def gather_logprobs(logits, targets):
    """log softmax(logits)[target] without materialising the full log-softmax."""
    logits = logits.float()
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)


def causal_forward(model, input_ids, attention_mask, start=1):
    """Run a causal LM and return (logits, hidden) for the positions that predict tokens
    start..L-1, i.e. hidden positions start-1..L-2. Identical to model(...).logits[:, start-1:-1]
    but only applies the LM head where needed (big memory saving when start = prompt width)."""
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids_from_mask(attention_mask),
                         use_cache=False).last_hidden_state[:, start - 1:-1]
    return model.lm_head(hidden), hidden


def token_logprobs(model, input_ids, attention_mask, start=1):
    """log π(x_t | x_<t) for every t ≥ start. Default start=1 → shape (B, L-1), column j = token j+1.
    PPO/eval call it with start=P (prompt width) to get the generated tokens only → (B, T)."""
    logits, _ = causal_forward(model, input_ids, attention_mask, start)
    return gather_logprobs(logits, input_ids[:, start:])


def entropy_from_logits(logits):
    """H(π(·|s)) = logsumexp(z) − Σ softmax(z)·z, per position."""
    logits = logits.float()
    return torch.logsumexp(logits, -1) - (torch.softmax(logits, -1) * logits).sum(-1)


def summary_mask(attention_mask, prompt_lens):
    """(B, L-1), aligned with token_logprobs: 1 exactly on summary tokens (incl. the
    final EOS), 0 on prompt and padding. For right-padded collate_sft/collate_pref batches."""
    L = attention_mask.size(1)
    t = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)       # absolute target position
    is_summary = t >= prompt_lens.to(attention_mask.device).unsqueeze(1)
    return (is_summary & attention_mask[:, 1:].bool()).float()


def masked_sum(x, mask):
    return (x * mask).sum()


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def sequence_logprob(model, ids, mask, prompt_lens):
    """log π(y | x) = Σ over summary tokens of log π(y_t | x, y_<t)."""
    return (token_logprobs(model, ids, mask) * summary_mask(mask, prompt_lens)).sum(1)


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, tok, input_ids, attention_mask, sample=False, max_new_tokens=MAX_SUMMARY_TOKENS):
    """Sampling settings fixed by the spec (§2.4 4.1); greedy otherwise (§2.2, §2.6)."""
    kwargs = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    return model.generate(input_ids=input_ids, attention_mask=attention_mask,
                          max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                          eos_token_id=tok.eos_token_id, **kwargs)


def response_mask(resp, eos_id):
    """1 up to and including the first EOS, 0 afterwards (everything after it is ignored).
    Returns (mask (B,T) long, index of the last kept token (B,))."""
    T = resp.size(1)
    is_eos = resp == eos_id
    first_eos = is_eos.long().argmax(1)
    last = torch.where(is_eos.any(1), first_eos, torch.full_like(first_eos, T - 1))
    mask = (torch.arange(T, device=resp.device).unsqueeze(0) <= last.unsqueeze(1)).long()
    return mask, last


@torch.no_grad()
def rollout(model, tok, prompt_ids, prompt_mask, sample):
    """Generate from left-padded prompts and return everything later stages need:
    full sequence, its attention mask, prompt width P, response mask and last index.
    Tokens after the first EOS are overwritten with EOS and masked out."""
    gen = generate(model, tok, prompt_ids, prompt_mask, sample=sample)
    P = prompt_ids.size(1)
    resp = gen[:, P:]
    rmask, last = response_mask(resp, tok.eos_token_id)
    resp = torch.where(rmask.bool(), resp, torch.full_like(resp, tok.eos_token_id))
    seq = torch.cat([prompt_ids, resp], 1)
    full_mask = torch.cat([prompt_mask, rmask], 1)
    return {"seq": seq, "mask": full_mask, "P": P, "resp": resp, "rmask": rmask, "last": last}


def response_logprobs(model, seq, mask, P):
    """log π(a_t | s_t) for the generated tokens only → (B, T)."""
    return token_logprobs(model, seq, mask, start=P)


def decode_responses(tok, resp, rmask):
    return [tok.decode(r[m.bool()], skip_special_tokens=True).strip() for r, m in zip(resp, rmask)]


# ----------------------------------------------------------------------------- logging & checks
class JsonlLogger:
    """Append-only JSON-lines log in results/logs/<stage>.jsonl."""

    def __init__(self, stage, resume=False):
        self.path = os.path.join(LOGS, f"{stage}.jsonl")
        if not resume:
            open(self.path, "w").close()
        self.t0 = time.time()

    def log(self, **row):
        row["elapsed_s"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)


def read_log(stage):
    with open(os.path.join(LOGS, f"{stage}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_summary(stage, **summary):
    """Per-stage final numbers for Table 2."""
    summary.update(gpu=gpu_name(), amp=USE_AMP, smoke=SMOKE)
    with open(os.path.join(LOGS, f"{stage}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def read_summary(stage):
    path = os.path.join(LOGS, f"{stage}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_check(line):
    with open(os.path.join(RESULTS, "correctness_checks.txt"), "a") as f:
        f.write(line + "\n")
    print("[CHECK]", line, flush=True)


def plot_panels(stage, panels, out_name, x="step"):
    """panels: list of (title, [log keys]). One subplot each, exported ≥2400 px wide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_log(stage)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (title, keys) in zip(axes[0], panels):
        for key in keys:
            pts = [(r[x], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o" if len(xs) < 30 else None, ms=3, lw=1.6, label=key)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, out_name), dpi=max(200, int(np.ceil(2400 / (6 * n)))))
    plt.close(fig)


# ----------------------------------------------------------------------------- Task 1
def _stats(lengths):
    return {"mean": round(float(np.mean(lengths)), 1), "max": int(np.max(lengths))}


def task1():
    from functools import partial
    from torch.utils.data import DataLoader
    from batching import collate_sft

    tok = load_tokenizer()
    sft_train, sft_val = get_splits(tok, "sft")
    pref_train, pref_val = get_splits(tok, "pref")
    eval_posts = get_splits(tok, "eval")

    stats = {
        "sft_train": len(sft_train), "sft_val": len(sft_val),
        "pref_train": len(pref_train), "pref_val": len(pref_val),
        "eval_posts": len(eval_posts),
        "sft_train_prompt_tokens": _stats([len(e["prompt_ids"]) for e in sft_train]),
        "sft_train_summary_tokens_incl_eos": _stats([len(e["summary_ids"]) for e in sft_train]),
        "sft_train_summary_tokens_excl_eos": _stats([len(e["summary_ids"]) - 1 for e in sft_train]),
    }
    with open(os.path.join(LOGS, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    expected = (N_SFT_TRAIN, N_SFT_VAL, N_PREF_TRAIN, N_PREF_VAL, N_EVAL_POSTS)
    got = (len(sft_train), len(sft_val), len(pref_train), len(pref_val), len(eval_posts))
    assert got == expected, f"subset sizes {got} != {expected}"

    # summary-mask check (§3.3): decoded masked tokens == summary text + EOS
    loader = DataLoader(sft_train, batch_size=8, shuffle=False,
                        collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))
    b = next(iter(loader))
    m = summary_mask(b["attention_mask"], b["prompt_lens"])
    for i in range(b["input_ids"].size(0)):
        decoded = tok.decode(b["input_ids"][i, 1:][m[i].bool()], clean_up_tokenization_spaces=False)
        expected_text = sft_train[i]["summary"] + tok.eos_token
        assert decoded == expected_text, f"mask mismatch row {i}: {decoded!r} vs {expected_text!r}"
        if i < 3:
            write_check(f"[summary_mask] row {i}: {decoded!r}")
    write_check(f"[summary_mask] OK on {b['input_ids'].size(0)} rows: masked tokens decode to summary + EOS")


if __name__ == "__main__":
    task1()
