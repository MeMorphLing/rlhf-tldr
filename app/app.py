"""Task 8.1 — front end: paste a Reddit post, see the SFT, PPO and DPO summaries side by side,
each with its reward-model score. Loads the local checkpoints in checkpoints/{sft,ppo,dpo,reward}.

    python app/app.py            # http://127.0.0.1:7860
"""
import html
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402

from common import (CKPT, DEVICE, autocast, build_prompt, load_tokenizer, load_model, freeze,  # noqa: E402
                    rollout, response_logprobs, decode_responses)
from config import MAX_PROMPT_TOKENS  # noqa: E402
from train_reward import load_reward_model  # noqa: E402

POLICIES = (("sft", "SFT", "imitates human TL;DRs"),
            ("ppo", "PPO", "RL against the reward model, KL-anchored to SFT"),
            ("dpo", "DPO", "optimised directly on preference pairs"))

EXAMPLES = [
    ["personalfinance", "Should I pay off my car loan early or build an emergency fund first?",
     "I'm 24 and just started my first full-time job. I have a car loan with about $6,000 left at 5% interest "
     "and only $800 in savings. My parents say I should throw every spare dollar at the loan, but my coworker "
     "says I should get three months of expenses saved first. My rent and bills are about $1,400 a month and "
     "I can put roughly $700 a month toward one or the other. Which one makes more sense?"],
    ["relationships", "My [27F] best friend [28F] keeps cancelling plans last minute",
     "We've been friends since college. In the last two months she has cancelled on me five times, usually an "
     "hour before we're supposed to meet, and always with a vague excuse. When I brought it up she said I was "
     "being dramatic. I miss her but I'm starting to feel like I'm the only one trying. Should I keep reaching "
     "out or give her space?"],
]

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#masthead h1 {font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 600; font-size: 2.1rem; margin: 0;}
#masthead p {font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .82rem; opacity: .75; margin: .3rem 0 0;}
.card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 16px 18px; height: 100%;}
.card.best {border: 2px solid var(--color-accent);}
.card .tag {font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .75rem; letter-spacing: .08em;
            text-transform: uppercase; opacity: .7;}
.card .name {font-family: 'IBM Plex Serif', Georgia, serif; font-size: 1.5rem; font-weight: 600;}
.card .summary {font-size: 1.02rem; line-height: 1.55; margin: 12px 0 14px; min-height: 4.5em;}
.card .metrics {display: flex; gap: 18px; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .82rem;}
.card .metrics b {font-size: 1.05rem;}
"""

_STATE = {}


def models():
    """Load every checkpoint once (4 × 135M parameters; fine on CPU)."""
    if not _STATE:
        _STATE["tok"] = load_tokenizer(CKPT["sft"])
        _STATE["sft"] = freeze(load_model(CKPT["sft"]))
        _STATE["ppo"] = freeze(load_model(CKPT["ppo"]))
        _STATE["dpo"] = freeze(load_model(CKPT["dpo"]))
        _STATE["rm"] = load_reward_model()
    return _STATE


def fit_prompt(tok, subreddit, title, post):
    """Build the dataset-style prompt; trim the post body if it exceeds the 448-token limit."""
    prompt = build_prompt(subreddit, title, post)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    trimmed = False
    while len(ids) > MAX_PROMPT_TOKENS:
        body = tok(post, add_special_tokens=False)["input_ids"]
        post = tok.decode(body[:max(0, len(body) - (len(ids) - MAX_PROMPT_TOKENS) - 5)])
        prompt = build_prompt(subreddit, title, post)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        trimmed = True
    return ids, trimmed


@torch.no_grad()
def summarise(subreddit, title, post):
    if not post.strip():
        raise gr.Error("Paste a post first.")
    s = models()
    tok = s["tok"]
    ids, trimmed = fit_prompt(tok, subreddit or "AskReddit", title or "", post)
    prompt_ids = torch.tensor([ids], device=DEVICE)
    prompt_mask = torch.ones_like(prompt_ids)

    results = {}
    for key, _, _ in POLICIES:
        ro = rollout(s[key], tok, prompt_ids, prompt_mask, sample=False)
        with autocast():
            score = float(s["rm"](ro["seq"], ro["mask"]).float())
            kl = 0.0
            if key != "sft":
                lp = response_logprobs(s[key], ro["seq"], ro["mask"], ro["P"]).float()
                lp_ref = response_logprobs(s["sft"], ro["seq"], ro["mask"], ro["P"]).float()
                kl = float(((lp - lp_ref) * ro["rmask"]).sum())
        results[key] = {"text": decode_responses(tok, ro["resp"], ro["rmask"])[0], "score": score,
                        "length": int(ro["rmask"].sum()), "kl": kl}

    best = max(results, key=lambda k: results[k]["score"])
    cards = []
    for key, name, blurb in POLICIES:
        r = results[key]
        kl = "—" if key == "sft" else f"{r['kl']:.2f}"
        cards.append(
            f"<div class='card{' best' if key == best else ''}'>"
            f"<div class='tag'>{html.escape(blurb)}</div><div class='name'>{name}</div>"
            f"<div class='summary'>{html.escape(r['text']) or '<i>(empty)</i>'}</div>"
            f"<div class='metrics'><span>rφ <b>{r['score']:.3f}</b></span><span>len <b>{r['length']}</b></span>"
            f"<span>KL→SFT <b>{kl}</b></span></div></div>")
    note = (f"Prompt: {len(ids)} tokens{' (post trimmed to fit the 448-token limit)' if trimmed else ''}. "
            f"Greedy decoding, ≤48 new tokens. Highlighted card = highest reward-model score.")
    return (*cards, note)


def build_ui():
    with gr.Blocks(css=CSS, title="TL;DR · SFT vs PPO vs DPO", theme=gr.themes.Base()) as demo:
        gr.HTML("<div id='masthead'><h1>TL;DR, aligned three ways</h1>"
                "<p>SmolLM2-135M · supervised fine-tuning → reward model → PPO / DPO · "
                "reproducing Stiennon et al. (2020)</p></div>")
        with gr.Row():
            subreddit = gr.Textbox(label="Subreddit", placeholder="relationships", scale=1)
            title = gr.Textbox(label="Title", placeholder="Post title", scale=3)
        post = gr.Textbox(label="Post", lines=9, placeholder="Paste the body of a Reddit post…")
        go = gr.Button("Summarise", variant="primary")
        with gr.Row(equal_height=True):
            outs = [gr.HTML() for _ in POLICIES]
        note = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[subreddit, title, post])
        go.click(summarise, inputs=[subreddit, title, post], outputs=[*outs, note])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
