# Week 4 Capstone: Full RLHF Loop and the Alignment Tax

## Objective
Construct the complete "Classic" Reinforcement Learning from Human Feedback pipeline. Take a pre-trained Supervised Fine-Tuned (SFT) model, a fully trained Reward Model, and optimize the policy using the PPO implementation you built in Week 3. Finally, systematically evaluate the effects of the KL divergence penalty on capability retention.

## Specifications

### 1. Architectural Assembly
You must interleave the components built over the last three weeks:
1.  **Actor/Reference Policy:** Select an off-the-shelf instruction-tuned model, such as `Llama-3-8B-Instruct` or `Mistral-v0.2-Instruct` (Quantization via 4-bit/8-bit LoRA is strictly necessitated by memory limitations). 
2.  **Critic ($R_\phi$):** Integrate the preference scalar model you authored in Week 2 on the `hh-rlhf` dataset.
3.  **RL Orchestrator:** Adopt the `ppo_trainer` constructed in Week 3 to drive the generation loop and parameter updates.

### 2. The Multi-Epoch Tuning Loop
Provision a multi-epoch, batched sequence generator using a set of open-ended conversational prompts.
1.  **Environment Initialization:** Seed your data samplers. Evaluate pre-optimization perplexity over a validation subset.
2.  **Reward Maximization:** Over multiple gradient accumulation steps, maximize the output of $R_\phi$ applied conditionally over generated text.
3.  **Metrics Tracking:** Rigorously log parameter dynamics.
    *   Actor loss, Critic MSE
    *   Mean generation length (mode collapse frequently correlates with abnormally short or infinite looped generations)
    *   Raw vs. KL-Adjusted Reward
    *   Mean $D_{KL}(\pi_\theta \parallel \pi_{ref})$

### 3. Ablation: The KL-Coefficient Sweeps
Empirically map the Alignment Tax by executing parallel training traces, varying only the coefficient $\beta \in \{0.001, 0.05, 0.2\}$.
*   For the trace $\beta \to 0$, intercept the checkpoint immediately *prior* to numerical instability and qualitatively assess the gibberish utilized to exploit the Reward Model.
*   For $\beta = 0.2$, calculate the terminal reward scalar and verify if the language construct strictly bounds to the initial SFT properties.

### 4. Technical Deliverables
*   `train_rlhf_capstone.py`: An end-to-end execution script utilizing a robust training framework (e.g., 🤗 `accelerate` or DeepSpeed ZeRO-2/3) to manage multi-node/multi-GPU synchronization.
*   **Documentation:** A brief markdown analysis directly attributing empirical variances in natural language outputs back to your algorithmic PPO hyperparameter settings.

Good luck. Moving past toy prototypes into multi-billion parameter RL landscapes is where theory meets raw systems engineering constraint.
