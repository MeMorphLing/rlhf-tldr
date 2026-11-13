# Numbers for Task 7 (generated — interpret them in analysis.md)

## 7.1 Reward model
- Validation pairwise accuracy: 0.6220 on 1000 pairs (chance 0.5)
- Accuracy by length relationship:
  - chosen_longer: {'n': 559, 'acc': 0.7513}
  - chosen_shorter: {'n': 399, 'acc': 0.4336}
  - same_length: {'n': 42, 'acc': 0.6905}
  - length_diff_le_2: {'n': 204, 'acc': 0.598}
  - length_diff_gt_10: {'n': 302, 'acc': 0.6325}
  - reward_vs_length_corr: 0.4712
- Most confident mistakes: `results/logs/reward_error_analysis.json`

## 7.2 / 7.4 PPO trends (mean of first vs last 10 iterations)
| metric | first 10 | last 10 |
|---|---|---|
| score_mean | -0.4784 | 0.1963 |
| kl_mean | 0.9718 | 10.4358 |
| length_mean | 38.2750 | 39.8625 |
| eos_rate | 0.8125 | 0.8125 |
| clip_frac | 0.0283 | 0.0210 |
| clip_frac_last_epoch | 0.0852 | 0.0631 |
| entropy | 2.9069 | 2.9674 |
| value_loss | 0.4945 | 0.2094 |
- Max clip fraction in any iteration: 0.0917
- First-epoch clip fraction: always 0.0000

## 7.2 Reward vs quality on the eval set
- SFT: rφ -0.753, ROUGE-L 0.2580, length 30.4
- PPO: rφ 0.221, ROUGE-L 0.2543, length 34.4
- DPO: rφ 0.178, ROUGE-L 0.2472, length 39.2

## 7.5 Reward model vs DPO implicit reward (same validation pairs)
- Reward model accuracy: 0.6220
- DPO implicit reward accuracy: 0.6100
- Both correct: 0.4930; agree on the decision: 0.7540

## 7.6 Cost
- sft: 18.3 min on Tesla T4
- reward: 38.6 min on Tesla T4
- ppo: 22.6 min on Tesla T4, models in GPU memory: 3, peak memory 11.09 GB
- dpo: 55.8 min on Tesla T4, models in GPU memory: 2
