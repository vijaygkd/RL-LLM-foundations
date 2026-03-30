# Week 3 Assignment: Sentiment Steering with PPO

## Objective
Implement Proximal Policy Optimization (PPO) to steer the generated completions of a generic language model (`gpt2`) toward generating explicitly positive movie reviews. You will utilize a pre-trained sentiment classifier as a proxy for the human preference scalar reward.

## Conceptual Architecture

### 1. Model Provisioning
*   **Actor Policy ($\pi_\theta$):** Use `gpt2` (or a very small causual LM). This is the model being optimized.
*   **Reference Policy ($\pi_{ref}$):** A frozen copy of the initial `gpt2` weights. Used to bound policy updates via the KL-divergence penalty.
*   **Reward Model ($R_\phi$):** A frozen, pre-trained sentiment analysis classification model (e.g., `distilbert-base-uncased-finetuned-sst-2-english` or similar) that outputs logits representing "positive sentiment".
*   **Value Critic ($V_\omega$):** A trainable critic that mirrors the hidden dimensionality of the actor body, terminating in a linear projection `nn.Linear(hidden_size, 1)` to predict state-values $V(s)$.

### 2. The RL Loop Iterations
Your training loop must execute the true RL formulation for language:
1.  **Generate:** Sample completions $y \sim \pi_\theta(\cdot \mid x)$ conditioned on a batch of initial sentiment-neutral prompts $x$ (e.g., "The movie was", "I thought the acting").
2.  **Evaluate:** Pass the concatenated sequence $[x, y]$ through the frozen Reward Model $R_\phi$ to obtain a terminal scalar score.
3.  **Log-Probs Computation:** Forward pass the sequences through both $\pi_\theta$ and the frozen $\pi_{ref}$ to compute conditional log probabilities for the *generated tokens* only.
4.  **Compute Rewards:** Calculate the timestep-level rewards incorporating the $\beta$-scaled KL penalty: $r_t = \beta ( \log \pi_{ref}(y_t) - \log \pi_\theta(y_t) )$ for $t < T$, and $r_T = R(x,y) + \beta (\dots)$.
5.  **Compute Returns & Advantages:** Utilize the Value Critic's scalar predictions to calculate Generalized Advantage Estimation (GAE) $\hat{A}_t$ and returns for your generated batch.
6.  **PPO Update:** Calculate the PPO Clipped Surrogate Objective ratio $r_t(\theta)$ and optimize the combined Actor loss and Critic Value MSE loss over several micro-epochs.

### 3. Deliverables
*   `ppo_trainer.py`: A fully functional script demonstrating the generation loop, advantage computation, and the clipped policy gradient update.
*   `tests/test_ppo.py`: Unit validations ensuring your KL penalty correctly penalizes distribution divergence, and that GAE calculations map appropriately to sequence lengths.
*   A logging output showing an upward trend in the reward scalar while maintaining coherent english syntactical structure.

## Technical Pointers
*   **Tensor Shapes:** Pay meticulous attention to your tensor shapes during shifting operations for log-probabilities. The reward corresponds to $y_t$, but the prediction for it occurred at state $s_{t-1}$.
*   **KL Crash:** If the KL divergence immediately crashes to $0$, your actor is likely not updating due to detached gradients. If it explodes to infinity, your sequence is completely degenerate (mode collapse).
*   **Truncation:** Keep maximum generation length constrained (e.g., 16-32 tokens) to streamline memory requirements and debugging iteration speed.

Good luck mapping the continuous optimization mathematics back to discrete vocabulary distributions.
