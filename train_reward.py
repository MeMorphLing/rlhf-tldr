"""Task 3 — Stage 2: reward model (PDF §2.3).

rφ(x, y) = Linear(h_last), where h_last is the final hidden state of the last
non-padding token of prompt + summary, from the SFT transformer body (LM head dropped).
L(φ) = − E[ log σ(rφ(x, y_w) − rφ(x, y_l)) ]          (lecture notes Eq. 40, Bradley–Terry)
"""
import json
import os
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CKPT, LOGS, SMOKE, DEVICE, autocast, make_scaler, optimizer_step, seed_all,
                    torch_generator, smoke, get_splits, load_tokenizer, load_model,
                    position_ids_from_mask, last_real_index, stack_pairs, JsonlLogger, read_log,
                    write_summary, write_check, plot_panels)
from batching import collate_pref

# ---- spec constants (§2.3) — do not change ----
LR = 1e-5
BATCH_PAIRS = 8
EPOCHS = 1
GRAD_CLIP = None           # not specified for Stage 2 → no clipping (norm still logged)
LOG_EVERY = 2 if SMOKE else 250
MICRO_PAIRS = int(os.environ.get("MICRO_BATCH", BATCH_PAIRS))
RM_FILE = "reward_model.pt"


class RewardModel(nn.Module):
    """SFT transformer body (model.model) + nn.Linear(hidden_size, 1)."""

    def __init__(self, base_path=CKPT["sft"], device=DEVICE):
        super().__init__()
        self.body = load_model(base_path, device).model          # LM head dropped
        self.head = nn.Linear(self.body.config.hidden_size, 1, device=device)
        nn.init.zeros_(self.head.weight)                          # starts indifferent: r_w − r_l = 0
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids_from_mask(attention_mask),
                           use_cache=False).last_hidden_state
        idx = last_real_index(attention_mask)
        h_last = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.head(h_last.float()).squeeze(-1)


def load_reward_model(path=CKPT["reward"], base_path=CKPT["sft"], device=DEVICE):
    """Rebuild the architecture from the SFT checkpoint, then load the trained weights."""
    rm = RewardModel(base_path, device)
    rm.load_state_dict(torch.load(os.path.join(path, RM_FILE), map_location=device))
    rm.eval()
    rm.requires_grad_(False)
    return rm


def pair_rewards(rm, batch, pad_id):
    """Scores for chosen and rejected in a single forward pass."""
    ids, mask, _ = stack_pairs(batch, pad_id)
    with autocast():
        r = rm(ids.to(DEVICE), mask.to(DEVICE))
    n = batch["chosen_ids"].size(0)
    return r[:n], r[n:]


@torch.inference_mode()
def evaluate(rm, loader, pad_id, return_rows=False):
    rm.eval()
    rw_all, rl_all = [], []
    for b in loader:
        rw, rl = pair_rewards(rm, b, pad_id)
        rw_all.append(rw.float().cpu())
        rl_all.append(rl.float().cpu())
    rm.train()
    rw, rl = torch.cat(rw_all), torch.cat(rl_all)
    out = {"val_acc": float((rw > rl).float().mean()), "val_loss": float(-F.logsigmoid(rw - rl).mean())}
    return (out, rw, rl) if return_rows else out


def padding_check(rm, loader, tok):
    """§3.3: the reward model must read the last *real* token (here: the summary's EOS)."""
    b = next(iter(loader))
    ids, mask, _ = stack_pairs(b, tok.pad_token_id)
    idx = last_real_index(mask)
    read = ids[torch.arange(ids.size(0)), idx]
    assert torch.all(read == tok.eos_token_id), "reward model is not reading the summary's EOS"
    assert torch.all(mask[torch.arange(ids.size(0)), idx] == 1), "reward model reads a padding position"
    write_check(f"[padding] reward model reads the final real token (summary EOS) for all {ids.size(0)} "
                f"right-padded rows; index found via attention_mask")


def error_analysis(val, rw, rl, n_show=20):
    """Data for Task 7.1: accuracy by length relationship + the most confident mistakes."""
    rows = []
    for ex, a, b in zip(val, rw.tolist(), rl.tolist()):
        lc, lr = len(ex["chosen_ids"]) - 1, len(ex["rejected_ids"]) - 1
        rows.append({"prompt": ex["prompt"], "chosen": ex["chosen"].strip(), "rejected": ex["rejected"].strip(),
                     "r_chosen": round(a, 4), "r_rejected": round(b, 4), "margin": round(a - b, 4),
                     "correct": a > b, "len_chosen": lc, "len_rejected": lr})

    def acc(sub):
        return {"n": len(sub), "acc": round(sum(r["correct"] for r in sub) / max(len(sub), 1), 4)}

    by_length = {
        "chosen_longer": acc([r for r in rows if r["len_chosen"] > r["len_rejected"]]),
        "chosen_shorter": acc([r for r in rows if r["len_chosen"] < r["len_rejected"]]),
        "same_length": acc([r for r in rows if r["len_chosen"] == r["len_rejected"]]),
        "length_diff_le_2": acc([r for r in rows if abs(r["len_chosen"] - r["len_rejected"]) <= 2]),
        "length_diff_gt_10": acc([r for r in rows if abs(r["len_chosen"] - r["len_rejected"]) > 10]),
        "reward_vs_length_corr": round(float(torch.corrcoef(torch.tensor(
            [[r["r_chosen"] for r in rows] + [r["r_rejected"] for r in rows],
             [r["len_chosen"] for r in rows] + [r["len_rejected"] for r in rows]], dtype=torch.float))[0, 1]), 4),
    }
    wrong = sorted([r for r in rows if not r["correct"]], key=lambda r: r["margin"])[:n_show]
    with open(os.path.join(LOGS, "reward_error_analysis.json"), "w") as f:
        json.dump({"by_length": by_length, "most_confident_mistakes": wrong}, f, indent=2)
    with open(os.path.join(LOGS, "reward_val_predictions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("r_chosen", "r_rejected", "correct", "len_chosen",
                                                  "len_rejected")}) + "\n")
    return by_length


def main():
    seed_all()
    tok = load_tokenizer()
    pad = tok.pad_token_id
    train, val = get_splits(tok, "pref")
    train, val = smoke(train, 32), smoke(val, 16)

    collate = partial(collate_pref, pad_id=pad)
    train_loader = DataLoader(train, batch_size=BATCH_PAIRS, shuffle=True, collate_fn=collate,
                              generator=torch_generator())
    val_loader = DataLoader(val, batch_size=16, shuffle=False, collate_fn=collate)

    rm = RewardModel(CKPT["sft"])
    rm.train()
    padding_check(rm, val_loader, tok)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=LR)
    scaler = make_scaler()
    logger = JsonlLogger("reward")
    logger.log(step=0, train_loss=None, train_acc=None, **evaluate(rm, val_loader, pad))

    t0, step, win_loss, win_acc = time.time(), 0, [], []
    for _ in range(EPOCHS):
        for batch in train_loader:
            step += 1
            n = batch["chosen_ids"].size(0)
            step_loss, step_correct = 0.0, 0
            for rows in torch.arange(n).split(MICRO_PAIRS):
                sub = {k: v[rows] for k, v in batch.items()}
                rw, rl = pair_rewards(rm, sub, pad)
                loss = -F.logsigmoid(rw - rl).sum() / n
                scaler.scale(loss).backward()
                step_loss += loss.item()
                step_correct += int((rw > rl).sum())
            grad_norm = optimizer_step(optimizer, scaler, rm.parameters(), GRAD_CLIP)
            win_loss.append(step_loss)
            win_acc.append(step_correct / n)
            if step % LOG_EVERY == 0 or step == len(train_loader) * EPOCHS:
                logger.log(step=step, train_loss=sum(win_loss) / len(win_loss),
                           train_acc=sum(win_acc) / len(win_acc), grad_norm=grad_norm,
                           **evaluate(rm, val_loader, pad))
                win_loss, win_acc = [], []
    train_minutes = (time.time() - t0) / 60

    os.makedirs(CKPT["reward"], exist_ok=True)
    torch.save(rm.state_dict(), os.path.join(CKPT["reward"], RM_FILE))
    tok.save_pretrained(CKPT["reward"])

    final, rw, rl = evaluate(rm, val_loader, pad, return_rows=True)
    by_length = error_analysis(val, rw, rl)
    last = [r for r in read_log("reward") if r["train_loss"] is not None][-1]
    write_summary("reward", steps=step, final_train_loss=last["train_loss"], val_pairwise_acc=final["val_acc"],
                  val_loss=final["val_loss"], n_val_pairs=len(val), accuracy_by_length=by_length,
                  train_minutes=round(train_minutes, 1))
    write_check(f"[reward] val pairwise accuracy {final['val_acc']:.4f} on {len(val)} pairs (chance = 0.5)")
    plot_panels("reward", [("Bradley–Terry loss", ["train_loss", "val_loss"]),
                           ("Pairwise accuracy", ["train_acc", "val_acc"])], "fig2_reward.png")


if __name__ == "__main__":
    main()
