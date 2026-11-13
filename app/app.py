"""Task 8.1 — front end: paste a Reddit post, see the SFT, PPO and DPO summaries side by side,
each with its reward-model score. Loads the local checkpoints in checkpoints/{sft,ppo,dpo,reward}.

    python app/app.py            # http://127.0.0.1:7860

Examples = your own cases in app/examples.json + real held-out test posts from
results/eval_summaries.jsonl (posts no model saw in training; their human reference is shown too).
"""

import html
import json
import os
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from common import (
    CKPT,
    DEVICE,
    autocast,
    build_prompt,
    load_tokenizer,
    load_model,
    freeze,  # noqa: E402
    rollout,
    response_logprobs,
    response_mask,
    decode_responses,
)
from config import MAX_PROMPT_TOKENS, MAX_SUMMARY_TOKENS  # noqa: E402
from train_reward import load_reward_model  # noqa: E402

POLICIES = (
    ("sft", "SFT", "Imitates human TL;DRs"),
    ("ppo", "PPO", "RL on the reward model, KL-leashed to SFT"),
    ("dpo", "DPO", "Optimized directly on preference pairs"),
)

CUSTOM_FILE = os.path.join(ROOT, "app", "examples.json")
EVAL_FILE = os.path.join(ROOT, "results", "eval_summaries.jsonl")
METRICS_FILE = os.path.join(ROOT, "results", "logs", "eval_metrics.json")
EVAL_IDS = [0, 25, 50, 100, 150, 200]
_PROMPT = re.compile(
    r"SUBREDDIT:\s*r/(.*?)\s*\nTITLE:\s*(.*?)\s*\nPOST:\s*(.*?)\s*\nTL;DR:\s*$", re.S
)


# ============================================================================ examples
def load_examples():
    """Returns (rows for gr.Examples, {post text: human reference summary})."""
    rows, refs = [], {}
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE, encoding="utf-8") as f:
            for e in json.load(f):
                rows.append([e["subreddit"], e["title"], e["post"]])
                if e.get("reference"):
                    refs[e["post"].strip()] = e["reference"]
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        for i in EVAL_IDS:
            m = _PROMPT.match(records[i]["prompt"]) if i < len(records) else None
            if m:
                rows.append(list(m.groups()))
                refs[m.group(3).strip()] = records[i]["reference"]
    return rows, refs


EXAMPLES, REFERENCES = load_examples()


def eval_table():
    """Table 3 from the evaluation run, if it is available locally."""
    if not os.path.exists(METRICS_FILE):
        return "_Run `python evaluate.py run` to show the held-out results here._"
    with open(METRICS_FILE) as f:
        m = json.load(f)
    lines = [
        f"Results on **{m['n_posts']} held-out posts** (greedy decoding):\n",
        "| Model | Mean rφ | Win rate vs SFT | KL to SFT | Length | ROUGE-L |",
        "|---|---|---|---|---|---|",
    ]
    for k, name, _ in POLICIES:
        x = m[k]
        win = "–" if k == "sft" else f"{x['win_rate_vs_sft']:.1%}"
        kl = "0" if k == "sft" else f"{x['kl_to_sft']:.2f}"
        lines.append(
            f"| {name} | {x['mean_score']:.3f} | {win} | {kl} | {x['length']:.1f} | {x['rougeL']:.3f} |"
        )
    return "\n".join(lines)


# ============================================================================ models
_STATE = {}


def models():
    """Load every checkpoint once (4 × 135M parameters; fine on CPU)."""
    if not _STATE:
        t0 = time.time()
        _STATE["tok"] = load_tokenizer(CKPT["sft"])
        for key, _, _ in POLICIES:
            _STATE[key] = freeze(load_model(CKPT[key]))
        _STATE["rm"] = load_reward_model()
        print(f"models loaded on {DEVICE} in {time.time() - t0:.1f}s", flush=True)
    return _STATE


def fit_prompt(tok, subreddit, title, post):
    """Dataset-style prompt; the post body is trimmed if the prompt exceeds the 448-token limit."""
    ids = tok(build_prompt(subreddit, title, post), add_special_tokens=False)[
        "input_ids"
    ]
    trimmed = False
    while len(ids) > MAX_PROMPT_TOKENS:
        body = tok(post, add_special_tokens=False)["input_ids"]
        post = tok.decode(
            body[: max(0, len(body) - (len(ids) - MAX_PROMPT_TOKENS) - 5)]
        )
        ids = tok(build_prompt(subreddit, title, post), add_special_tokens=False)[
            "input_ids"
        ]
        trimmed = True
    return ids, trimmed


@torch.no_grad()
def run_policy(s, key, prompt_ids, prompt_mask, sample, temperature, seed):
    """One summary from one policy + its rφ score + KL to SFT on that summary."""
    tok = s["tok"]
    t0 = time.time()
    if sample:
        torch.manual_seed(int(seed))
        gen = s[key].generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            do_sample=True,
            temperature=float(temperature),
            top_k=0,
            top_p=1.0,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        P = prompt_ids.size(1)
        rmask, _ = response_mask(gen[:, P:], tok.eos_token_id)
        resp = torch.where(
            rmask.bool(), gen[:, P:], torch.full_like(gen[:, P:], tok.eos_token_id)
        )
        ro = {
            "seq": torch.cat([prompt_ids, resp], 1),
            "mask": torch.cat([prompt_mask, rmask], 1),
            "P": P,
            "resp": resp,
            "rmask": rmask,
        }
    else:
        ro = rollout(s[key], tok, prompt_ids, prompt_mask, sample=False)
    with autocast():
        score = float(s["rm"](ro["seq"], ro["mask"]).float())
        kl = None
        if key != "sft":
            lp = response_logprobs(s[key], ro["seq"], ro["mask"], ro["P"]).float()
            lp_ref = response_logprobs(s["sft"], ro["seq"], ro["mask"], ro["P"]).float()
            kl = float(((lp - lp_ref) * ro["rmask"]).sum())
    length = int(ro["rmask"].sum())
    capped = (
        int(ro["resp"][0, length - 1]) != tok.eos_token_id
    )  # no EOS → ran into the 48-token cap
    return {
        "text": decode_responses(tok, ro["resp"], ro["rmask"])[0],
        "score": score,
        "kl": kl,
        "length": length,
        "capped": capped,
        "ms": int((time.time() - t0) * 1000),
    }


# ============================================================================ rendering
def placeholder_card(name, blurb):
    return (
        f"<div class='card'><div class='tag'>{html.escape(blurb)}</div><div class='name'>{name}</div>"
        f"<div class='summary muted'>Summary appears here.</div></div>"
    )


def render_card(name, blurb, r, best, lo, hi, rouge=None):
    width = 8 + 92 * ((r["score"] - lo) / (hi - lo) if hi > lo else 1.0)
    kl = "—" if r["kl"] is None else f"{r['kl']:.2f}"
    badge = "<span class='badge'>highest rφ</span>" if best else ""
    cap = "<span class='warn'>· hit the 48-token cap</span>" if r["capped"] else ""
    rouge_html = "" if rouge is None else f"<span>ROUGE-L <b>{rouge:.3f}</b></span>"
    return (
        f"<div class='card{' best' if best else ''}'>"
        f"<div class='tag'>{html.escape(blurb)}</div>"
        f"<div class='name'>{name}{badge}</div>"
        f"<div class='summary'>{html.escape(r['text']) or '<i>(empty)</i>'}</div>"
        f"<div class='bar'><div style='width:{width:.0f}%'></div></div>"
        f"<div class='metrics'><span>rφ <b>{r['score']:.3f}</b></span><span>len <b>{r['length']}</b></span>"
        f"<span>KL→SFT <b>{kl}</b></span>{rouge_html}</div>"
        f"<div class='foot'>{r['ms']} ms {cap}</div></div>"
    )


def render_reference(ref):
    if not ref:
        return ""
    return (
        f"<div class='ref'><span class='tag'>Human reference TL;DR · held-out test post</span>"
        f"<div>{html.escape(ref.strip())}</div></div>"
    )


# ============================================================================ handlers
def summarise(subreddit, title, post, mode, temperature, seed):
    if not post or not post.strip():
        raise gr.Error("Paste a post first.")
    s = models()
    ids, trimmed = fit_prompt(s["tok"], subreddit or "AskReddit", title or "", post)
    prompt_ids = torch.tensor([ids], device=DEVICE)
    prompt_mask = torch.ones_like(prompt_ids)
    sample = mode.startswith("Sample")

    results = {
        k: run_policy(s, k, prompt_ids, prompt_mask, sample, temperature, seed)
        for k, _, _ in POLICIES
    }

    reference = REFERENCES.get(post.strip())
    rouges = {}
    if reference:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouges = {
            k: scorer.score(reference.strip(), r["text"])["rougeL"].fmeasure
            for k, r in results.items()
        }

    scores = [r["score"] for r in results.values()]
    lo, hi = min(scores) - 0.25, max(scores)
    best = max(results, key=lambda k: results[k]["score"])
    cards = [
        render_card(name, blurb, results[k], k == best, lo, hi, rouges.get(k))
        for k, name, blurb in POLICIES
    ]
    decoding = (
        "greedy (as in the evaluation)"
        if not sample
        else f"sampling, T={temperature}, seed={int(seed)}"
    )
    note = (
        f"Prompt: {len(ids)} tokens{' — post trimmed to fit the 448-token limit' if trimmed else ''} · "
        f"decoding: {decoding} · ≤{MAX_SUMMARY_TOKENS} new tokens · device: {DEVICE}.  \n"
        f"**rφ** = the trained reward model's score (higher = preferred). "
        f"**KL→SFT** = how far each model's own summary drifts from the SFT model."
    )
    return (*cards, render_reference(reference), note)


def clear():
    return ("", "", "", *[placeholder_card(n, b) for _, n, b in POLICIES], "", "")


# ============================================================================ UI
CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#masthead h1 {font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 600; font-size: 2.2rem; margin: 0;}
#masthead p {font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .82rem; opacity: .75; margin: .35rem 0 0;}
.card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px 18px; height: 100%;
       background: var(--block-background-fill);}
.card.best {border: 2px solid var(--color-accent);
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-accent) 15%, transparent);}
.card .tag, .ref .tag {font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .72rem; letter-spacing: .07em;
       text-transform: uppercase; opacity: .65;}
.card .name {font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.5rem; font-weight: 600; display: flex;
       align-items: center; gap: 10px;}
.card .badge {font-family: 'IBM Plex Mono', monospace; font-size: .68rem; font-weight: 500; padding: 2px 8px;
       border-radius: 99px; background: var(--color-accent); color: white;}
.card .summary {font-size: 1.02rem; line-height: 1.55; margin: 12px 0 14px; min-height: 4.6em;}
.card .summary.muted {opacity: .45; font-style: italic;}
.card .bar {height: 6px; border-radius: 99px; background: var(--border-color-primary); overflow: hidden; margin-bottom: 10px;}
.card .bar div {height: 100%; background: var(--color-accent); border-radius: 99px;}
.card .metrics {display: flex; flex-wrap: wrap; gap: 6px 16px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
       font-size: .8rem;}
.card .metrics b {font-size: 1rem;}
.card .foot {margin-top: 8px; font-family: 'IBM Plex Mono', monospace; font-size: .72rem; opacity: .6;}
.card .warn {color: #d97706;}
.ref {border-left: 4px solid var(--color-accent); padding: 10px 16px; border-radius: 6px;
      background: var(--block-background-fill); font-size: 1rem; line-height: 1.5;}
"""


def build_ui():
    with gr.Blocks(
        css=CSS, title="TL;DR · SFT vs PPO vs DPO", theme=gr.themes.Base()
    ) as demo:
        gr.HTML(
            "<div id='masthead'><h1>TL;DR, aligned three ways</h1>"
            "<p>SmolLM2-135M · supervised fine-tuning → reward model → PPO / DPO · "
            "reproducing Stiennon et al. (2020)</p></div>"
        )
        with gr.Row():
            subreddit = gr.Textbox(
                label="Subreddit", placeholder="relationships", scale=1
            )
            title = gr.Textbox(label="Title", placeholder="Post title", scale=3)
        post = gr.Textbox(
            label="Post", lines=8, placeholder="Paste the body of a Reddit post…"
        )
        with gr.Accordion("Decoding options", open=False):
            mode = gr.Radio(
                ["Greedy (matches the evaluation)", "Sample"],
                value="Greedy (matches the evaluation)",
                label="Decoding",
            )
            with gr.Row():
                temperature = gr.Slider(
                    0.3, 1.5, value=0.7, step=0.1, label="Temperature (sampling only)"
                )
                seed = gr.Number(value=42, precision=0, label="Seed (sampling only)")
        with gr.Row():
            go = gr.Button("Summarize", variant="primary", scale=4)
            reset = gr.Button("Clear", scale=1)
        with gr.Row(equal_height=True):
            outs = [gr.HTML(placeholder_card(n, b)) for _, n, b in POLICIES]
        ref = gr.HTML()
        note = gr.Markdown()
        if EXAMPLES:
            gr.Examples(
                EXAMPLES,
                inputs=[subreddit, title, post],
                examples_per_page=10,
                label="Examples — your own cases first, then held-out test posts no model saw",
            )
        with gr.Accordion("Held-out evaluation results", open=False):
            gr.Markdown(eval_table())
        go.click(
            summarise,
            inputs=[subreddit, title, post, mode, temperature, seed],
            outputs=[*outs, ref, note],
        )
        reset.click(clear, outputs=[subreddit, title, post, *outs, ref, note])
    return demo


if __name__ == "__main__":
    models()  # load before the first click
    build_ui().queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
    )
