# Task 7 — Analysis

> Fill every section from your own runs. The numbers come from `results/analysis_data.md`
> (`make analysis`), `results/tables.md`, the figures in `results/figures/`, and the human evaluation.
> Delete these quoted guidance lines before submitting.

## 7.1 How accurate is the reward model, and what does it get wrong?

> Report the validation pairwise accuracy on the 1,000 pairs against the 50% chance level. Use the
> accuracy-by-length breakdown and `reward_vs_length_corr` to say whether the model leans on length.
> Show three wrong pairs from `results/logs/reward_error_analysis.json` (post excerpt, chosen, rejected,
> both scores). For each one, say why it is hard: near-duplicate summaries, the labeller preferring the
> shorter one, a factual detail only the post can settle, or a noisy label.

## 7.2 PPO training: score, KL, length — is a higher rφ a better summary?

> Embed `figures/fig3_ppo.png`. Compare the first 10 and last 10 iterations for score, KL and length.
> Compare Table 3 (rφ, ROUGE-L, length) with Table 4 (which summaries the raters actually preferred).
> If rφ went up while ROUGE-L or the human picks did not, name it: the policy optimises the reward
> model, not the human reference (Goodhart / reward over-optimisation).

## 7.3 The role of β_KL (reasoning only, not run)

> Using Eq. 41, r_t = −β_KL (log πθ − log πref) + 1[t = T] rφ: β_KL prices every token's deviation from
> SFT. With β_KL = 0 the objective is pure rφ maximisation with no anchor, so expect: unbounded KL growth,
> exploitation of rφ's blind spots (for example length or stock phrases), loss of fluency and diversity
> (entropy collapse), and summaries that score high but read badly. Relate this to what your curves show
> at β_KL = 0.05.

## 7.4 Clip fraction

> Epoch 1 is exactly 0 by construction (r_t = 1). Report the typical and maximum clip fraction over
> epochs 2–4 (`clip_frac_last_epoch`). A small value means each update stays inside the [1 − ε, 1 + ε]
> trust region, so updates are small and stable. A value that rises over training means the policy is
> moving further per update.

## 7.5 DPO: where did the reward go?

> Eqs. 55–58: the KL-constrained optimum is π*(y|x) = πref(y|x) exp(r(x,y)/β) / Z(x), so
> r(x,y) = β log π*(y|x)/πref(y|x) + β log Z(x). Substituted into Bradley–Terry, the β log Z(x) term
> appears for both y_w and y_l and cancels in the difference, so Z(x) is never computed. The policy
> itself is the reward model. Compare DPO's implicit reward accuracy with the reward model's accuracy
> on the same 1,000 pairs, and their agreement rate (from `analysis_data.md`).

## 7.6 Cost, and which method to choose

> PPO: 3 models in GPU memory (policy + value head, frozen πref, frozen rφ) plus generation every
> iteration. DPO: 2 models (policy, frozen πref), no generation, no reward model. Report the minutes and
> peak memory for each stage from Table 2 / `analysis_data.md`, then give your recommendation and the
> reason for it.
