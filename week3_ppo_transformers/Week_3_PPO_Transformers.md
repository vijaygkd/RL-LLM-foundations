# Week 3: PPO for Transformers & The TRL Loop

## Overview
This week explores the final stage of "Classic" RLHF: optimizing a Pre-Trained Language Model (the Actor) against a trained Scalar Reward Model using Proximal Policy Optimization (PPO).

## Key Concepts

### 1. The Four-Model Setup
A standard PPO setup for LLM alignment requires maintaining four distinct networks in memory simultaneously. Managing this overhead is a major engineering challenge:
1.  **Actor Model ($\pi_\theta$):** The trainable language model policy we are updating.
2.  **Reference Model ($\pi_{ref}$):** A frozen copy of the initial SFT model used to calculate the KL-divergence penalty.
3.  **Reward Model ($R_\phi$):** A frozen text classifier trained (typically via Bradley-Terry) to predict human preference scores.
4.  **Value Model ($V_\omega$):** A trainable critic that predicts the expected future sum of rewards from a given state. It matches the Actor's base architecture but projects the final hidden states to a scalar head.

### 2. The PPO Objective with KL-Penalty
If left unconstrained, PPO will "reward hack"—aggressively degrading fluency and grammar to maximize the scalar reward. We regularize this behavior via a KL penalty:

$$ R(x, y) = r_\phi(x, y) - \beta D_{KL}[\pi_\theta(y | x) \parallel \pi_{ref}(y | x)] $$

Where $\beta$ controls the strength of the alignment tax. Formally, this KL penalty is injected token-by-token:
$$ r_t = \text{KL}_{t} = -\log \left( \frac{\pi_\theta(a_t | s_t)}{\pi_{ref}(a_t | s_t)} \right) = \log \pi_{ref}(a_t | s_t) - \log \pi_\theta(a_t | s_t) $$
The final discrete token generation step receives the absolute score $r_\phi$, while all intervening tokens accrue intermediate KL penalties.

### 3. Generalized Advantage Estimation (GAE)
To perform actor updates, we compute the advantage of each token action. Because language steps are discrete, we use the GAE estimator to balance variance and bias over sequence generation:
$$ \delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t) $$
$$ \hat{A}_t = \sum_{l=0}^{T-t}(\gamma \lambda)^l \delta_{t+l} $$

### 4. The Clipped Surrogate Objective
We update the policy by optimizing the expected advantage, applying a clip to prevent destructively large catastrophic policy updates in a single iteration step:
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ] $$
where $r_t(\theta)$ is the probability ratio: $\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$.

## Reading List
- [Transformer Reinforcement Learning (TRL) Documentation](https://huggingface.co/docs/trl/index)
- *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
- *Learning to summarize from human feedback* (Stiennon et al., 2020)
