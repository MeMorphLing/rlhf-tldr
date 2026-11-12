"""Task 5 — Stage 4: DPO (PDF §2.5).

L_DPO(θ) = − E[ log σ( β log πθ(y_w|x)/πref(y_w|x) − β log πθ(y_l|x)/πref(y_l|x) ) ]   (Eq. 60)
log π(y|x) is the sum of token log-probs over the summary tokens only.
"""
import json
import math
import os
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CKPT, LOGS, SMOKE, DEVICE, autocast, make_scaler, optimizer_step, seed_all,
                    torch_generator, smoke, get_splits, load_tokenizer, load_model, freeze, assert_frozen,
                    save_policy, sequence_logprob, stack_pairs, JsonlLogger, read_log, write_summary,
                    write_check, plot_panels)
from batching import collate_pref

# ---- spec constants (§2.5) — do not change ----
BETA = 0.1
LR = 2e-6
BATCH_PAIRS = 8
EPOCHS = 1
GRAD_CLIP = None           # not specified for Stage 4 → no clipping (norm still logged)
LOG_EVERY = 2 if SMOKE else 250
MICRO_PAIRS = int(os.environ.get("MICRO_BATCH", BATCH_PAIRS))


def dpo_logits(policy, ref, batch, pad_id, grad=True):
    """Returns z = β[(log πθ(y_w) − log πref(y_w)) − (log πθ(y_l) − log πref(y_l))] and the two
    implicit rewards. Chosen and rejected go through each model in one forward pass."""
    ids, mask, plens = (t.to(DEVICE) for t in stack_pairs(batch, pad_id))
    n = batch["chosen_ids"].size(0)
    with torch.no_grad(), autocast():
        ref_lp = sequence_logprob(ref, ids, mask, plens)
    with torch.set_grad_enabled(grad), autocast():
        pol_lp = sequence_logprob(policy, ids, mask, plens)
    reward_w = BETA * (pol_lp[:n] - ref_lp[:n])
    reward_l = BETA * (pol_lp[n:] - ref_lp[n:])
    return reward_w - reward_l, reward_w, reward_l


@torch.inference_mode()
def evaluate(policy, ref, loader, pad_id, return_rows=False):
    policy.eval()
    zs, rws, rls = [], [], []
    for b in loader:
        z, rw, rl = dpo_logits(policy, ref, b, pad_id, grad=False)
        zs.append(z.float().cpu()); rws.append(rw.float().cpu()); rls.append(rl.float().cpu())
    policy.train()
    z, rw, rl = torch.cat(zs), torch.cat(rws), torch.cat(rls)
    out = {"val_loss": float(-F.logsigmoid(z).mean()), "val_implicit_acc": float((z > 0).float().mean()),
           "val_reward_chosen": float(rw.mean()), "val_reward_rejected": float(rl.mean()),
           "val_margin": float(z.mean())}
    return (out, z) if return_rows else out


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

    policy = load_model(CKPT["sft"])
    policy.train()
    ref = freeze(load_model(CKPT["sft"]))
    assert_frozen("DPO reference πref", ref)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    scaler = make_scaler()
    logger = JsonlLogger("dpo")
    logger.log(step=0, train_loss=None, train_implicit_acc=None, **evaluate(policy, ref, val_loader, pad))

    t0, step, win_loss, win_acc = time.time(), 0, [], []
    for _ in range(EPOCHS):
        for batch in train_loader:
            step += 1
            n = batch["chosen_ids"].size(0)
            step_loss, step_correct = 0.0, 0
            for rows in torch.arange(n).split(MICRO_PAIRS):
                z, _, _ = dpo_logits(policy, ref, {k: v[rows] for k, v in batch.items()}, pad)
                loss = -F.logsigmoid(z.float()).sum() / n
                scaler.scale(loss).backward()
                step_loss += loss.item()
                step_correct += int((z > 0).sum())
            if step == 1:
                assert abs(step_loss - math.log(2)) < 1e-4, f"first DPO loss {step_loss} != log 2"
                write_check(f"[dpo] first-step loss = {step_loss:.6f} (log 2 = {math.log(2):.6f}) because πθ = πref")
            grad_norm = optimizer_step(optimizer, scaler, policy.parameters(), GRAD_CLIP)
            win_loss.append(step_loss)
            win_acc.append(step_correct / n)
            if step % LOG_EVERY == 0 or step == len(train_loader) * EPOCHS:
                logger.log(step=step, train_loss=sum(win_loss) / len(win_loss),
                           train_implicit_acc=sum(win_acc) / len(win_acc), grad_norm=grad_norm,
                           **evaluate(policy, ref, val_loader, pad))
                win_loss, win_acc = [], []
    train_minutes = (time.time() - t0) / 60

    save_policy(policy, tok, CKPT["dpo"])
    final, z = evaluate(policy, ref, val_loader, pad, return_rows=True)
    with open(os.path.join(LOGS, "dpo_val_predictions.jsonl"), "w") as f:
        for v in z.tolist():
            f.write(json.dumps({"z": round(v, 5), "correct": v > 0}) + "\n")

    last = [r for r in read_log("dpo") if r["train_loss"] is not None][-1]
    write_summary("dpo", steps=step, final_train_loss=last["train_loss"],
                  val_implicit_acc=final["val_implicit_acc"], val_loss=final["val_loss"], n_val_pairs=len(val),
                  models_in_gpu_memory=2, train_minutes=round(train_minutes, 1))
    write_check(f"[dpo] val implicit reward accuracy {final['val_implicit_acc']:.4f} on {len(val)} pairs")
    plot_panels("dpo", [("DPO loss", ["train_loss", "val_loss"]),
                        ("Implicit reward accuracy", ["train_implicit_acc", "val_implicit_acc"]),
                        ("Implicit rewards β·log πθ/πref", ["val_reward_chosen", "val_reward_rejected"])],
                "fig4_dpo.png")


if __name__ == "__main__":
    main()
