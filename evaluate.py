"""Task 6 — compare SFT, PPO and DPO on the 300 held-out evaluation posts (PDF §2.6, §3, §4).

Subcommands
  run            greedy summaries for all three policies + every metric in §3.1 → eval_summaries.jsonl, tables.md
  human-sheet    blind A/B/C sheets for 30 posts (§3.2), one independently shuffled sheet per rater
  human-score    merge the filled sheets → human_eval.csv + Table 4
  samples        10 qualitative samples (§4.6) → samples.md
  analysis-data  numbers needed for Task 7 → results/analysis_data.md
  tables         rebuild results/tables.md from the logs
  figures        rebuild figures 1–4 from the logs
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from common import (CKPT, LOGS, RESULTS, SEED, DEVICE, autocast, seed_all, smoke, get_splits,
                    load_tokenizer, load_model, freeze, rollout, response_logprobs, decode_responses,
                    read_log, read_summary, plot_panels)
from batching import collate_prompts

POLICIES = ("sft", "ppo", "dpo")
GEN_BATCH = 16
N_HUMAN = 30
N_BOOT = 2000
EVAL_FILE = os.path.join(RESULTS, "eval_summaries.jsonl")
METRICS_FILE = os.path.join(LOGS, "eval_metrics.json")


# ============================================================================ run
@torch.no_grad()
def generate_and_score(policy, sft_ref, rm, tok, posts):
    """Greedy summaries (≤48 new tokens) + rφ score + log πθ(y|x) − log πSFT(y|x) on the policy's own summary."""
    out = []
    for i in range(0, len(posts), GEN_BATCH):
        chunk = posts[i:i + GEN_BATCH]
        pb = collate_prompts(chunk, tok.pad_token_id)
        ro = rollout(policy, tok, pb["input_ids"].to(DEVICE), pb["attention_mask"].to(DEVICE), sample=False)
        m = ro["rmask"].float()
        with autocast():
            lp_pol = response_logprobs(policy, ro["seq"], ro["mask"], ro["P"]).float()
            lp_sft = response_logprobs(sft_ref, ro["seq"], ro["mask"], ro["P"]).float()
            score = rm(ro["seq"], ro["mask"]).float()
        kl = ((lp_pol - lp_sft) * m).sum(1)
        texts = decode_responses(tok, ro["resp"], ro["rmask"])
        ended = ro["resp"][torch.arange(len(chunk), device=DEVICE), ro["last"]] == tok.eos_token_id
        for j, text in enumerate(texts):
            out.append({"summary": text, "score": float(score[j]), "kl_to_sft": float(kl[j]),
                        "length": int(m[j].sum()), "ended_with_eos": bool(ended[j])})
    return out


def bootstrap_ci(values, n=N_BOOT, seed=SEED):
    """95% percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(records):
    metrics = {}
    for name in POLICIES:
        scores = [r[name]["score"] for r in records]
        m = {"mean_score": float(np.mean(scores)), "mean_score_ci95": bootstrap_ci(scores),
             "kl_to_sft": float(np.mean([r[name]["kl_to_sft"] for r in records])),
             "length": float(np.mean([r[name]["length"] for r in records])),
             "rougeL": float(np.mean([r[name]["rougeL"] for r in records])),
             "rougeL_ci95": bootstrap_ci([r[name]["rougeL"] for r in records]),
             "eos_rate": float(np.mean([r[name]["ended_with_eos"] for r in records]))}
        if name != "sft":
            wins = [float(r[name]["score"] > r["sft"]["score"]) for r in records]
            m.update(win_rate_vs_sft=float(np.mean(wins)), win_rate_vs_sft_ci95=bootstrap_ci(wins),
                     ties_with_sft=float(np.mean([r[name]["score"] == r["sft"]["score"] for r in records])),
                     identical_text_to_sft=float(np.mean([r[name]["summary"] == r["sft"]["summary"]
                                                          for r in records])))
        metrics[name] = m
    metrics["ppo_vs_dpo_win_rate"] = float(np.mean([r["ppo"]["score"] > r["dpo"]["score"] for r in records]))
    metrics["n_posts"] = len(records)
    return metrics


def cmd_run(_):
    from rouge_score import rouge_scorer
    from train_reward import load_reward_model

    seed_all()
    tok = load_tokenizer()
    posts = smoke(get_splits(tok, "eval"), 16)
    rm = load_reward_model()
    sft = freeze(load_model(CKPT["sft"]))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    results = {}
    for name in POLICIES:
        policy = sft if name == "sft" else freeze(load_model(CKPT[name]))
        results[name] = generate_and_score(policy, sft, rm, tok, posts)
        if name != "sft":
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"{name}: done")

    records = []
    for i, post in enumerate(posts):
        ref = post["summary"].strip()
        rec = {"idx": i, "prompt": post["prompt"], "reference": ref}
        for name in POLICIES:
            r = dict(results[name][i])
            r["rougeL"] = scorer.score(ref, r["summary"])["rougeL"].fmeasure
            rec[name] = r
        records.append(rec)
    with open(EVAL_FILE, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    metrics = summarise(records)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    assert abs(metrics["sft"]["kl_to_sft"]) < 1e-3, "SFT KL to itself must be 0"
    cmd_tables(None)
    print(json.dumps(metrics, indent=2))


# ============================================================================ tables
def _fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def cmd_tables(_):
    lines = ["# Results tables\n", "Generated by `python evaluate.py tables` from `results/logs/`.\n"]

    stats_path = os.path.join(LOGS, "data_stats.json")
    if os.path.exists(stats_path):
        s = json.load(open(stats_path))
        p, q = s["sft_train_prompt_tokens"], s["sft_train_summary_tokens_incl_eos"]
        q2 = s["sft_train_summary_tokens_excl_eos"]
        lines += ["## Table 1 – Data\n", "| | |", "|---|---|",
                  f"| SFT train / validation pairs | {s['sft_train']:,} / {s['sft_val']:,} |",
                  f"| Preference train / validation pairs | {s['pref_train']:,} / {s['pref_val']:,} |",
                  f"| Evaluation posts | {s['eval_posts']} |",
                  f"| Mean / max prompt length, SFT train (tokens) | {p['mean']} / {p['max']} |",
                  f"| Mean / max summary length, SFT train (tokens, incl. EOS) | {q['mean']} / {q['max']} |",
                  f"| Mean / max summary length, SFT train (tokens, excl. EOS) | {q2['mean']} / {q2['max']} |", ""]

    sft, rew, ppo, dpo = (read_summary(k) for k in ("sft", "reward", "ppo", "dpo"))
    if any((sft, rew, ppo, dpo)):
        def t(sm):
            return "–" if not sm else f"{sm['train_minutes']} min / {sm['gpu']}"
        lines += ["## Table 2 – Training\n", "| Stage | Final train loss | Validation metric | Time / GPU |",
                  "|---|---|---|---|"]
        if sft:
            lines.append(f"| SFT | {_fmt(sft['final_train_loss'])} | val. loss: {_fmt(sft['final_val_loss'])} | {t(sft)} |")
        if rew:
            lines.append(f"| Reward model | {_fmt(rew['final_train_loss'])} | pairwise acc.: "
                         f"{_fmt(rew['val_pairwise_acc'])} | {t(rew)} |")
        if ppo:
            lines.append(f"| PPO | {_fmt(ppo['final_total_loss_last10'])}¹ | final mean rφ: "
                         f"{_fmt(ppo['final_mean_score_last10'])}² | {t(ppo)} |")
        if dpo:
            lines.append(f"| DPO | {_fmt(dpo['final_train_loss'])} | implicit reward acc.: "
                         f"{_fmt(dpo['val_implicit_acc'])} | {t(dpo)} |")
        lines += ["", "Train losses are averaged over the last logging window (250 steps). "
                  "¹ PPO total loss −L_CLIP + c_V·L_V − c_H·H, mean of the last 10 iterations. "
                  "² Mean rollout rφ over the last 10 iterations (sampled, temperature 1).", ""]

    if os.path.exists(METRICS_FILE):
        m = json.load(open(METRICS_FILE))
        lines += [f"## Table 3 – Final comparison ({m['n_posts']} evaluation posts, greedy decoding)\n",
                  "| Policy | Mean rφ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |", "|---|---|---|---|---|---|"]
        for name in POLICIES:
            x = m[name]
            win = "–" if name == "sft" else f"{x['win_rate_vs_sft']:.1%}"
            kl = "0" if name == "sft" else f"{x['kl_to_sft']:.2f}"
            lines.append(f"| {name.upper()} | {x['mean_score']:.3f} | {win} | {kl} | {x['length']:.1f} | "
                         f"{x['rougeL']:.4f} |")
        lines += ["", "95% bootstrap CIs (2,000 resamples): " + "; ".join(
            f"{n.upper()} rφ [{m[n]['mean_score_ci95'][0]:.3f}, {m[n]['mean_score_ci95'][1]:.3f}], "
            f"ROUGE-L [{m[n]['rougeL_ci95'][0]:.4f}, {m[n]['rougeL_ci95'][1]:.4f}]" for n in POLICIES) + ".",
            "Win rate = share of posts where the method's summary has a strictly higher rφ than SFT's. "
            "Length in tokens incl. EOS. ROUGE-L = F-measure vs. the human reference (rouge-score, Porter stemmer).", ""]

    h_path = os.path.join(LOGS, "human_eval_summary.json")
    if os.path.exists(h_path):
        h = json.load(open(h_path))
        lines += [f"## Table 4 – Human evaluation ({h['n_posts']} posts)\n",
                  "| | SFT picked | PPO picked | DPO picked |", "|---|---|---|---|"]
        for label, counts in h["counts"].items():
            lines.append(f"| {label} | {counts['sft']} | {counts['ppo']} | {counts['dpo']} |")
        if h.get("agreement") is not None:
            lines.append(f"| Agreement between raters | {h['agreement']} / {h['n_posts']} | | |")
        lines += ["", h.get("note", ""), ""]

    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote results/tables.md")


# ============================================================================ human evaluation
def _human_posts(records):
    return sorted(random.Random(SEED).sample(range(len(records)), min(N_HUMAN, len(records))))


def cmd_human_sheet(args):
    records = [json.loads(l) for l in open(EVAL_FILE)]
    chosen = _human_posts(records)
    key = {}
    for r in range(1, args.raters + 1):
        rng = random.Random(SEED + 1000 * r)
        rows, key[f"rater{r}"] = [], []
        for item, idx in enumerate(chosen, 1):
            order = list(POLICIES)
            rng.shuffle(order)
            rec = records[idx]
            rows.append({"item": item, "post": rec["prompt"],
                         "A": rec[order[0]]["summary"], "B": rec[order[1]]["summary"],
                         "C": rec[order[2]]["summary"], "pick (A/B/C)": ""})
            key[f"rater{r}"].append({"item": item, "post_index": idx, "A": order[0], "B": order[1], "C": order[2]})
        path = os.path.join(RESULTS, f"human_eval_blind_r{r}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {path}")
    with open(os.path.join(RESULTS, "human_eval_key.json"), "w") as f:
        json.dump(key, f, indent=2)
    print("wrote results/human_eval_key.json — do not open it before every sheet is filled in")


def cmd_human_score(args):
    key = json.load(open(os.path.join(RESULTS, "human_eval_key.json")))
    raters = sorted(key)
    labels = args.labels or [f"Rater {i}" for i in range(1, len(raters) + 1)]
    assert len(labels) == len(raters), f"need {len(raters)} labels"
    picks = {}
    for r, rater in enumerate(raters, 1):
        rows = list(csv.DictReader(open(os.path.join(RESULTS, f"human_eval_blind_r{r}.csv"))))
        mapping = {k["item"]: k for k in key[rater]}
        picks[rater] = {}
        for row in rows:
            p = row["pick (A/B/C)"].strip().upper()
            assert p in ("A", "B", "C"), f"{rater} item {row['item']}: pick must be A, B or C (got {p!r})"
            k = mapping[int(row["item"])]
            picks[rater][k["post_index"]] = k[p]

    post_ids = sorted(picks[raters[0]])
    with open(os.path.join(RESULTS, "human_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["post_index"] + [f"{l} pick" for l in labels] + (["agree"] if len(raters) == 2 else []))
        for idx in post_ids:
            row = [idx] + [picks[r][idx] for r in raters]
            if len(raters) == 2:
                row.append(int(row[1] == row[2]))
            w.writerow(row)

    counts = {l: {p: sum(v == p for v in picks[r].values()) for p in POLICIES} for l, r in zip(labels, raters)}
    agreement = sum(picks[raters[0]][i] == picks[raters[1]][i] for i in post_ids) if len(raters) == 2 else None
    summary = {"n_posts": len(post_ids), "counts": counts, "agreement": agreement, "note": args.note or ""}
    with open(os.path.join(LOGS, "human_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    cmd_tables(None)


# ============================================================================ qualitative samples
def cmd_samples(args):
    records = [json.loads(l) for l in open(EVAL_FILE)]
    if args.ids:
        picked = [(i, "selected") for i in args.ids]
    else:
        # candidates: biggest ROUGE-L gains / losses of PPO or DPO vs SFT (where the text actually differs),
        # then fill with random posts. Read them and keep only clear cases.
        def delta(r):
            return max(r["ppo"]["rougeL"], r["dpo"]["rougeL"]) - r["sft"]["rougeL"], \
                   min(r["ppo"]["rougeL"], r["dpo"]["rougeL"]) - r["sft"]["rougeL"]
        diff = [r for r in records if r["ppo"]["summary"] != r["sft"]["summary"]
                or r["dpo"]["summary"] != r["sft"]["summary"]]
        better = sorted(diff, key=lambda r: -delta(r)[0])[:3]
        worse = sorted([r for r in diff if r not in better], key=lambda r: delta(r)[1])[:3]
        used = {r["idx"] for r in better + worse}
        rest = [r for r in records if r["idx"] not in used]
        fill = random.Random(SEED).sample(rest, 10 - len(used))
        picked = [(r["idx"], "candidate: PPO/DPO better (ROUGE-L)") for r in better] + \
                 [(r["idx"], "candidate: PPO/DPO worse (ROUGE-L)") for r in worse] + \
                 [(r["idx"], "random") for r in fill]
    lines = ["# Qualitative samples (10 evaluation posts)\n",
             "Each post shows the human reference and the greedy SFT, PPO and DPO summaries with their rφ score "
             "and ROUGE-L. The *Why* line is the author's judgement after reading the post.\n"]
    for n, (idx, tag) in enumerate(picked, 1):
        r = records[idx]
        lines += [f"## Sample {n} — eval post #{idx} ({tag})\n", "```", r["prompt"], "```\n",
                  f"- **Reference:** {r['reference']}"]
        for name in POLICIES:
            x = r[name]
            lines.append(f"- **{name.upper()}** (rφ {x['score']:.2f}, ROUGE-L {x['rougeL']:.3f}, "
                         f"{x['length']} tok): {x['summary']}")
        lines += ["", "**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, "
                      "hallucination)._", ""]
    with open(os.path.join(RESULTS, "samples.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote results/samples.md — read each post and fill in the *Why* lines")


# ============================================================================ Task 7 numbers
def cmd_analysis_data(_):
    out = ["# Numbers for Task 7 (generated — interpret them in analysis.md)\n"]
    rew = read_summary("reward")
    if rew:
        out += ["## 7.1 Reward model", f"- Validation pairwise accuracy: {rew['val_pairwise_acc']:.4f} "
                f"on {rew['n_val_pairs']} pairs (chance 0.5)", "- Accuracy by length relationship:"]
        out += [f"  - {k}: {v}" for k, v in rew["accuracy_by_length"].items()]
        out += ["- Most confident mistakes: `results/logs/reward_error_analysis.json`", ""]
    if os.path.exists(os.path.join(LOGS, "ppo.jsonl")):
        rows = read_log("ppo")
        k = min(10, len(rows))
        avg = lambda key, rs: sum(r[key] for r in rs) / len(rs)
        out += ["## 7.2 / 7.4 PPO trends (mean of first vs last 10 iterations)",
                "| metric | first 10 | last 10 |", "|---|---|---|"]
        for key in ("score_mean", "kl_mean", "length_mean", "eos_rate", "clip_frac", "clip_frac_last_epoch",
                    "entropy", "value_loss"):
            out.append(f"| {key} | {avg(key, rows[:k]):.4f} | {avg(key, rows[-k:]):.4f} |")
        out += [f"- Max clip fraction in any iteration: {max(r['clip_frac'] for r in rows):.4f}",
                f"- First-epoch clip fraction: always {max(r['clip_frac_epoch1'] for r in rows):.4f}", ""]
    if os.path.exists(METRICS_FILE):
        m = json.load(open(METRICS_FILE))
        out += ["## 7.2 Reward vs quality on the eval set"] + [
            f"- {n.upper()}: rφ {m[n]['mean_score']:.3f}, ROUGE-L {m[n]['rougeL']:.4f}, length {m[n]['length']:.1f}"
            for n in POLICIES] + [""]
    rm_p, dpo_p = os.path.join(LOGS, "reward_val_predictions.jsonl"), os.path.join(LOGS, "dpo_val_predictions.jsonl")
    if os.path.exists(rm_p) and os.path.exists(dpo_p):
        a = [json.loads(l)["correct"] for l in open(rm_p)]
        b = [json.loads(l)["correct"] for l in open(dpo_p)]
        out += ["## 7.5 Reward model vs DPO implicit reward (same validation pairs)",
                f"- Reward model accuracy: {np.mean(a):.4f}", f"- DPO implicit reward accuracy: {np.mean(b):.4f}",
                f"- Both correct: {np.mean([x and y for x, y in zip(a, b)]):.4f}; agree on the decision: "
                f"{np.mean([x == y for x, y in zip(a, b)]):.4f}", ""]
    out += ["## 7.6 Cost"]
    for stage in ("sft", "reward", "ppo", "dpo"):
        s = read_summary(stage)
        if s:
            extra = f", models in GPU memory: {s['models_in_gpu_memory']}" if "models_in_gpu_memory" in s else ""
            extra += f", peak memory {s['peak_gpu_mem_gb']} GB" if "peak_gpu_mem_gb" in s else ""
            out.append(f"- {stage}: {s['train_minutes']} min on {s['gpu']}{extra}")
    with open(os.path.join(RESULTS, "analysis_data.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


def cmd_figures(_):
    plot_panels("sft", [("SFT loss (summary tokens only)", ["train_loss", "val_loss"])], "fig1_sft_loss.png")
    plot_panels("reward", [("Bradley–Terry loss", ["train_loss", "val_loss"]),
                           ("Pairwise accuracy", ["train_acc", "val_acc"])], "fig2_reward.png")
    plot_panels("ppo", [("Mean reward-model score rφ", ["score_mean"]), ("Mean KL to reference", ["kl_mean"]),
                        ("Mean summary length (tokens)", ["length_mean"]),
                        ("Clip fraction", ["clip_frac", "clip_frac_last_epoch"])], "fig3_ppo.png")
    plot_panels("ppo", [("Policy loss −L_CLIP", ["policy_loss"]), ("Value loss", ["value_loss"]),
                        ("Entropy", ["entropy"])], "fig3b_ppo_losses.png")
    plot_panels("dpo", [("DPO loss", ["train_loss", "val_loss"]),
                        ("Implicit reward accuracy", ["train_implicit_acc", "val_implicit_acc"]),
                        ("Implicit rewards β·log πθ/πref", ["val_reward_chosen", "val_reward_rejected"])],
                "fig4_dpo.png")
    print("figures written to results/figures/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    hs = sub.add_parser("human-sheet"); hs.add_argument("--raters", type=int, default=2)
    sc = sub.add_parser("human-score")
    sc.add_argument("--labels", nargs="+")
    sc.add_argument("--note", default="", help="how the raters were recruited (shown under Table 4)")
    sp = sub.add_parser("samples"); sp.add_argument("--ids", type=int, nargs="+")
    sub.add_parser("analysis-data"); sub.add_parser("tables"); sub.add_parser("figures")
    args = ap.parse_args()
    {"run": cmd_run, "human-sheet": cmd_human_sheet, "human-score": cmd_human_score, "samples": cmd_samples,
     "analysis-data": cmd_analysis_data, "tables": cmd_tables, "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main()
