# InstructGPT (Ouyang et al., 2022)

## 1. Core Oneliner & Motivation
**Oneliner:** Use comparison datasets, containing human preference labels for generations on the same prompt, to:
1. Train a Reward Model (RM) that maps `(prompt, output) -> scalar score`.
2. Train a policy using Proximal Policy Optimization (PPO) by using the RM for rating the model's completions.

**Why:** This helps align the language model with human intent. The goal is teaching models to follow instructions while remaining helpful, honest (truthful), and harmless (HHH).

## 2. Dataset Pipeline
The training pipeline requires three distinct datasets:

1. **SFT Dataset:** Bootstraps the model to follow instructions using a `(prompt, desired_behavior)` dataset to create the initial supervised policy.
2. **Comparison Dataset (RM Data):** Ask human labelers to rank multiple completions from the SFT (or on-policy) models generated from the same prompt. This creates the $y_w$ (chosen) vs $y_l$ (rejected) pairs.
3. **Prompt Dataset (RL Data):** Use the active on-policy model to generate rollout completions on a dataset of prompts, and use the trained RM to score them for PPO training.

## 3. Mathematical Objectives (Pseudocode)

### Reward Model (RM) Objective
For each comparison pair ($C_w$, $C_l$), run a forward pass with the RM to get a scalar output for both the chosen and rejected completions.

```python
# L_rm = - mean(log(sigmoid(r_w - r_l)))

def rm_loss(r_w, r_l):
    loss = -torch.mean(torch.log(torch.sigmoid(r_w - r_l)))
    return loss
```

### PPO Objective
For each completion pair (prompt, sequence) generated during the rollout, calculating the loss per-token:

```python
# RLHF PPO Objective
# L_ppo   = elementwise clipped(r_rm + adv_gae - beta_kl)
# L_ptx   = -mean(log_prob(target_token))
# L_total = L_ppo + gamma * L_ptx
```
*(Note: I expanded your GAE/KL notation slightly to reflect that the GAE advantage `adv_gae` is what actually gets clipped, and the KL divergence acts as a per-token negative reward prior to the GAE calculation).*
