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
For each completion pair (prompt, sequence) generated during the rollout, calculate the loss per-token.

**1. Calculate Advantages:**
The Reward Model scalar `r_rm` and the KL divergence penalty `beta_kl` are used to define the reward at each token $t$. These rewards are passed through Generalized Advantage Estimation (GAE) to give us our per-token advantages.
```python
# rewards_t = -beta * kl_t
# rewards_final = r_rm - beta * kl_final
# adv_gae = calculate_gae(rewards)
```

**2. The Clipped Surrogate Objective:**
The final PPO objective only multiplies the probability ratio by the pre-calculated advantage:
```python
# ratio = pi_new(token) / pi_old(token)
# L_ppo_clipped_objective = -mean( min(ratio * adv_gae, clipped_ratio * adv_gae) )
```

**3. Total Objective:**
Summing the PPO gradients with the Pretraining Mix (PTX):
```python
# L_ptx   = -mean(log_prob(target_token))
# L_total = L_ppo_clipped_objective + gamma * L_ptx
```
