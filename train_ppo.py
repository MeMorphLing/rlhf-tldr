"""Task 4 — Stage 3: PPO (PDF §2.4, lecture notes Sections 2–4).

Each generated token is an action a_t; the state s_t is the prompt plus the tokens generated so far.
  r_t   = −β_KL (log πθold(a_t|s_t) − log πref(a_t|s_t)) + 1[t = T] rφ(x, y)        (Eq. 41)
  R_t   = Σ_k γ^k r_{t+k},   A_t = R_t − Vθold(s_t), normalised over the batch      (Eqs. 19–20, 27)
  L(θ)  = −L^CLIP(θ) + c_V E[(Vθ(s_t) − R_t)²] − c_H E[H(πθ(·|s_t))]               (Eqs. 32, 34, 35)
Every expectation is a mean over the generated (non-padding) tokens of the batch.

Alignment: the hidden state / logits at absolute position P+t−1 predict a_t (at position P+t),
so log π(a_t|s_t) and V(s_t) are read from the same position (§2.4, 4.2).
"""
import json
import os
import random
import time

import torch
import torch.nn as nn

from common import (CKPT, ROOT, SEED, SMOKE, DEVICE, autocast, make_scaler, optimizer_step, seed_all,
                    get_splits, load_tokenizer, load_model, freeze, assert_frozen, save_policy,
                    causal_forward, gather_logprobs, entropy_from_logits, token_logprobs, last_real_index,
                    masked_sum, masked_mean, rollout, decode_responses, JsonlLogger, read_log,
                    write_summary, write_check, plot_panels)
from batching import collate_prompts
from train_reward import load_reward_model

# ---- spec constants (§2.4, 4.5) — do not change ----
CLIP_EPS = 0.2
BETA_KL = 0.05
C_V = 0.1
C_H = 0.01
GAMMA = 1.0
PPO_EPOCHS = 4
LR = 1e-5
GRAD_CLIP = 1.0
PROMPTS_PER_ROLLOUT = 16
ITERATIONS = 3 if SMOKE else 200
# ---- engineering knobs (do not affect the algorithm) ----
CHUNK = int(os.environ.get("MICRO_BATCH", PROMPTS_PER_ROLLOUT))  # rows per backward; 8 = "two halves"
SAVE_EVERY = 2 if SMOKE else 50
RESUME_PATH = os.path.join(ROOT, "checkpoints", "ppo_resume", "state.pt")
RATIO_TOL = 1e-3 if os.environ.get("RLHF_AMP", "1") == "1" and DEVICE == "cuda" else 1e-5


class PolicyWithValue(nn.Module):
    """πθ (the SFT causal LM) + Vθ, a new linear value head on the last hidden layer."""

    def __init__(self, path=CKPT["sft"], device=DEVICE):
        super().__init__()
        self.lm = load_model(path, device)
        self.value_head = nn.Linear(self.lm.config.hidden_size, 1, device=device)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, seq, mask, P):
        """For the generated tokens a_0..a_{T-1}: log πθ(a_t|s_t), Vθ(s_t), H(πθ(·|s_t)), each (B, T)."""
        logits, hidden = causal_forward(self.lm, seq, mask, start=P)
        logprobs = gather_logprobs(logits, seq[:, P:])
        values = self.value_head(hidden.float()).squeeze(-1)
        return logprobs, values, entropy_from_logits(logits)


def discounted_returns(rewards, mask, gamma=GAMMA):
    """R_t = r_t + γ R_{t+1}, computed right-to-left over the generated tokens only."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(0), device=rewards.device)
    for t in reversed(range(rewards.size(1))):
        running = rewards[:, t] + gamma * running * mask[:, t]
        returns[:, t] = running
    return returns * mask


def chunks(n):
    return torch.arange(n, device=DEVICE).split(CHUNK)


@torch.no_grad()
def make_experience(policy, ref, rm, tok, prompts):
    """§2.4 4.1–4.4: rollout, stored quantities (no gradients), per-token rewards, returns, advantages."""
    policy.eval()
    pb = collate_prompts(prompts, tok.pad_token_id)
    ro = rollout(policy.lm, tok, pb["input_ids"].to(DEVICE), pb["attention_mask"].to(DEVICE), sample=True)
    seq, full, P, m, last = ro["seq"], ro["mask"], ro["P"], ro["rmask"].float(), ro["last"]

    old_lp, old_v, ref_lp = [], [], []
    for rows in chunks(seq.size(0)):                      # same chunking as the update → exact ratio = 1
        with autocast():
            lp, v, _ = policy(seq[rows], full[rows], P)
            rlp = token_logprobs(ref, seq[rows], full[rows], start=P)
        old_lp.append(lp.float()); old_v.append(v.float()); ref_lp.append(rlp.float())
    old_lp, old_v, ref_lp = torch.cat(old_lp), torch.cat(old_v), torch.cat(ref_lp)
    with autocast():
        score = rm(seq, full).float()

    kl_per_token = (old_lp - ref_lp) * m
    rewards = -BETA_KL * kl_per_token
    rewards[torch.arange(seq.size(0), device=DEVICE), last] += score
    returns = discounted_returns(rewards, m)
    adv = (returns - old_v) * m
    mu = masked_mean(adv, m)
    std = masked_mean((adv - mu) ** 2, m).sqrt()
    adv = (adv - mu) / (std + 1e-8) * m

    return {"seq": seq, "mask": full, "P": P, "m": m, "last": last, "resp": ro["resp"],
            "old_lp": old_lp, "old_v": old_v, "ref_lp": ref_lp, "score": score,
            "kl_per_seq": kl_per_token.sum(1), "returns": returns, "adv": adv,
            "ended": ro["resp"][torch.arange(seq.size(0), device=DEVICE), last] == tok.eos_token_id}


def ppo_update(policy, optimizer, scaler, exp):
    """§2.4 4.5: K epochs over the same rollout, one optimiser step per epoch (full batch of 16)."""
    policy.eval()                    # SmolLM2 has no dropout; eval() keeps forward passes identical
    seq, full, P, m = exp["seq"], exp["mask"], exp["P"], exp["m"]
    n_tok = m.sum()
    epochs = []
    for epoch in range(PPO_EPOCHS):
        st = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "clipped": 0.0, "ratio_dev": 0.0, "approx_kl": 0.0}
        for rows in chunks(seq.size(0)):
            mr = m[rows]
            with autocast():
                lp, v, ent = policy(seq[rows], full[rows], P)
            ratio = torch.exp(lp - exp["old_lp"][rows])
            adv = exp["adv"][rows]
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv)
            l_clip = masked_sum(surrogate, mr) / n_tok
            l_value = masked_sum((v - exp["returns"][rows]) ** 2, mr) / n_tok
            l_ent = masked_sum(ent, mr) / n_tok
            loss = -l_clip + C_V * l_value - C_H * l_ent
            scaler.scale(loss).backward()

            with torch.no_grad():
                outside = ((ratio < 1 - CLIP_EPS) | (ratio > 1 + CLIP_EPS)).float()
                st["pg"] += float(-l_clip); st["vf"] += float(l_value); st["ent"] += float(l_ent)
                st["clipped"] += float(masked_sum(outside, mr))
                st["approx_kl"] += float(masked_sum(exp["old_lp"][rows] - lp, mr))
                st["ratio_dev"] = max(st["ratio_dev"], float(((ratio - 1).abs() * mr).max()))
        st["grad_norm"] = optimizer_step(optimizer, scaler, policy.parameters(), GRAD_CLIP)
        st["clip_frac"] = st.pop("clipped") / float(n_tok)
        st["approx_kl"] /= float(n_tok)
        epochs.append(st)
    return epochs


def save_resume(policy, optimizer, scaler, iteration):
    os.makedirs(os.path.dirname(RESUME_PATH), exist_ok=True)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "iteration": iteration, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               RESUME_PATH)


def main():
    seed_all()
    tok = load_tokenizer()
    train, _ = get_splits(tok, "sft")
    order = random.Random(SEED).sample(range(len(train)), ITERATIONS * PROMPTS_PER_ROLLOUT)  # 3,200 distinct

    policy = PolicyWithValue(CKPT["sft"])
    ref = freeze(load_model(CKPT["sft"]))
    rm = load_reward_model()
    assert_frozen("PPO reference πref", ref)
    assert_frozen("PPO reward model rφ", rm)
    write_check("[ppo] models in GPU memory: policy+value head (trainable), πref (frozen), rφ (frozen) = 3")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    scaler = make_scaler()
    start = 1
    resume = os.environ.get("RESUME", "0") == "1" and os.path.exists(RESUME_PATH)
    if resume:
        state = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        policy.load_state_dict(state["policy"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"]); torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["iteration"] + 1
        print(f"resuming from iteration {start}")
    logger = JsonlLogger("ppo", resume=resume)
    if resume:                                            # drop log rows written after the saved state
        kept = [r for r in read_log("ppo") if r["step"] < start]
        with open(logger.path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)

    max_ratio_dev_e1 = 0.0
    for it in range(start, ITERATIONS + 1):
        t_it = time.time()
        prompts = [train[i] for i in order[(it - 1) * PROMPTS_PER_ROLLOUT: it * PROMPTS_PER_ROLLOUT]]
        exp = make_experience(policy, ref, rm, tok, prompts)
        epochs = ppo_update(policy, optimizer, scaler, exp)

        e1 = epochs[0]
        max_ratio_dev_e1 = max(max_ratio_dev_e1, e1["ratio_dev"])
        assert e1["ratio_dev"] < RATIO_TOL and e1["clip_frac"] == 0.0, \
            f"iteration {it}: first-epoch ratio deviates by {e1['ratio_dev']:.2e} (clip frac {e1['clip_frac']})"
        kl_mean = float(exp["kl_per_seq"].mean())
        if it == 1:
            assert abs(kl_mean) < 1e-4, f"KL at iteration 1 is {kl_mean}, expected 0 (policy == SFT)"
            write_check(f"[ppo] iteration 1: mean KL to reference = {kl_mean:.2e} (policy still equals SFT)")
            seq, mask = exp["seq"], exp["mask"]
            read = seq[torch.arange(seq.size(0), device=DEVICE), last_real_index(mask)]
            expect = exp["resp"][torch.arange(seq.size(0), device=DEVICE), exp["last"]]
            assert torch.equal(read, expect)
            write_check("[padding] PPO prompts are left-padded for generate(); reward model reads the final real "
                        "token of every prompt+summary (found with attention_mask)")
            sample = decode_responses(tok, exp["resp"][:1], exp["m"][:1].long())[0]
            print(f"[ppo] iteration 1 sample summary: {sample!r}")

        mean = lambda k: sum(e[k] for e in epochs) / len(epochs)
        logger.log(step=it,
                   score_mean=float(exp["score"].mean()), score_std=float(exp["score"].std()),
                   kl_mean=kl_mean, length_mean=float(exp["m"].sum(1).mean()),
                   eos_rate=float(exp["ended"].float().mean()),
                   clip_frac=mean("clip_frac"), clip_frac_epoch1=e1["clip_frac"],
                   clip_frac_last_epoch=epochs[-1]["clip_frac"], ratio_dev_epoch1=e1["ratio_dev"],
                   policy_loss=mean("pg"), value_loss=mean("vf"), entropy=mean("ent"),
                   approx_kl_update=epochs[-1]["approx_kl"], grad_norm=mean("grad_norm"),
                   return_mean=float(masked_mean(exp["returns"], exp["m"])),
                   value_mean=float(masked_mean(exp["old_v"], exp["m"])),
                   iter_seconds=round(time.time() - t_it, 1),
                   gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0)
        if it % SAVE_EVERY == 0 and it < ITERATIONS:
            save_resume(policy, optimizer, scaler, it)

    save_policy(policy.lm, tok, CKPT["ppo"])                      # value head not saved (§2.4 4.7)
    write_check(f"[ppo] every iteration: first-epoch max |r_t(θ) − 1| = {max_ratio_dev_e1:.2e}, clip fraction = 0")

    rows = read_log("ppo")
    tail = rows[-min(10, len(rows)):]
    avg = lambda k, rs: sum(r[k] for r in rs) / len(rs)
    total_loss = avg("policy_loss", tail) + C_V * avg("value_loss", tail) - C_H * avg("entropy", tail)
    write_summary("ppo", iterations=ITERATIONS, final_mean_score_last10=avg("score_mean", tail),
                  first_mean_score=rows[0]["score_mean"], final_kl_last10=avg("kl_mean", tail),
                  final_length_last10=avg("length_mean", tail), first_length=rows[0]["length_mean"],
                  final_total_loss_last10=total_loss, models_in_gpu_memory=3,
                  peak_gpu_mem_gb=max(r["gpu_mem_gb"] for r in rows),
                  train_minutes=round(sum(r["iter_seconds"] for r in rows) / 60, 1))
    plot_panels("ppo", [("Mean reward-model score rφ", ["score_mean"]), ("Mean KL to reference", ["kl_mean"]),
                        ("Mean summary length (tokens)", ["length_mean"]),
                        ("Clip fraction", ["clip_frac", "clip_frac_last_epoch"])], "fig3_ppo.png", x="step")
    plot_panels("ppo", [("Policy loss −L_CLIP", ["policy_loss"]), ("Value loss", ["value_loss"]),
                        ("Entropy", ["entropy"])], "fig3b_ppo_losses.png", x="step")


if __name__ == "__main__":
    main()
