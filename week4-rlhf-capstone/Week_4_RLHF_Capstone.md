# Week 4: The Full RLHF Implementation Challenge

## Overview
This week marks the capstone of Month 1 (The PPO Foundations). We are departing from minimal environments (Gridworld, GPT-2 Sentiment Steering) to integrate the complete RLHF pipeline utilized to train large-scale alignment models like InstructGPT.

## Key Concepts: The "Alignment Tax"
When we apply a scalar reward signal to optimize a language model, the broad, high-entropy distribution cultivated during pre-training is artificially constricted. This inherently implies that while the policy becomes highly optimized along the target reward vector (e.g., helpfulness), its general capabilities, linguistic diversity, and perplexity on out-of-distribution tasks degrade. This capability degradation is formally termed the **Alignment Tax**.

This week, you will observe this phenomenon firsthand. By aggressively over-optimizing the PPO objective without sufficient regularization, you will witness mode collapse—where the policy discovers a pathological sub-manifold capable of perfectly exploiting vulnerabilities in the reward model (Reward Hacking) at the total expense of natural language coherence.

### The KL-Coefficient as the Tax Rate
The primary regularization mechanism to constrain the optimization landscape is $\beta$, the coefficient scaling the KL Divergence penalty:
$$ R(x, y) = r(x, y) - \beta D_{KL}(\pi_\theta(y | x) \parallel \pi_{ref}(y | x)) $$
Tuning this hyperparameter is an essential empirical skill for a research scientist. 
- $\beta \to \infty$: An overly conservative policy that stubbornly anchors to the SFT distribution, ignoring the reward gradient.
- $\beta \to 0$: Severe reward hacking and immediate distribution collapse.

## Reading List
*   *Training language models to follow instructions with human feedback* (InstructGPT - Ouyang et al., 2022) - Prioritize Sections 2 and 3 regarding the PPO training objective formulation.
*   *Scaling Laws for Reward Model Overoptimization* (Gao et al., 2022) - Detailed examination of proxy reward degradation.
