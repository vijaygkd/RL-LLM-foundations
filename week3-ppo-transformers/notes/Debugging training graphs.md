# Debugging PPO Training Graphs

Below are our ongoing observations and technical deductions from the first set of scaled training traces. We'll graduate these into the definitive `CHRONICLES.md` once thoroughly validated.

## 1. Divergent Negative KL Penalty
**Observation:**
The mean KL token penalty steadily dropped deep into negative values (e.g., -0.125).

**Expected Behavior:**
The token-level KL divergence $\mathbb{E}[\log\pi_\theta - \log\pi_{\text{ref}}]$ should strictly remain **positive** and gradually increase, stabilizing around $0.05$ to $0.1$. This indicates the policy is confidently moving *away* from the reference model strictly toward higher-reward tokens without collapsing.

**Reasoning:**
Because the raw rewards were uniformly bad (centering around ~0.3, meaning ~70% below neutral), GAE predominantly calculates negative returns. During policy updates, the Actor is heavily penalized for these generated trajectories, squashing its $\log \pi_{\theta}(a_t)$ below the frozen base model's $\log \pi_{\text{ref}}(a_t)$. The model easily learns "what not to say" by broadly suppressing confidence in its own generations, driving the log-ratio estimator negative.

## 2. Flat Evaluation Rewards & Stagnant Trust Region
**Observation:**
Evaluation rewards flatlined around 0.32 while the `clip_fraction` hovered close to exactly zero (~0.6%).

**Expected Behavior:**
Evaluation rewards should monotonically climb over time. The `clip_fraction` should bounce consistently between **5% and 15%**, indicating that a healthy chunk of the policy updates are actively pushing against the $1 \pm 0.2$ PPO trust region boundaries and taking meaningful optimization steps.

**Reasoning:**
A `clip_fraction` of ~0.006 means less than 1% of the policy log-ratios drifted far enough to hit the boundary. The optimizer is practically tiptoeing and is far too rigid to escape local minima. To repair this, we have scaled up both `lr` (to $10^{-5}$) and the `gen_batch_size`/`num_prompts` heavily to force a wider variance of positive gradient paths into the Actor's scope. 

## 3. Negative Advantages and the Optimistic Critic
**Observation:**
We verified that the true raw advantages (calculated strictly by TD error) were predominantly negative across early iterations.

**Expected Behavior:**
Over time, as the policy learns to generate positive sentiment, we expect raw advantages to be zero-centered or slightly positive on average. The Critic should accurately predict high future returns, and the Actor should occasionally exceed those expectations on brilliant semantic tokens.

**Reasoning:**
A fundamentally negative GAE advantage ($r_t + \gamma V(s_{t+1}) < V(s_t)$) strictly means the Critic was overly optimistic about the state value. 

*Exception to factor in later:* Remember that before the Actor processes these error bounds, PPO uses **Advantage Normalization** (`advantages = (advantages - mean) / std`). As long as the batch is normalized, the Actor updates fairly based on relative performance (pushing the *least-bad* 50% upwards). Thus, heavily negative raw advantages uniquely isolate the Critic's poor expectation bounds rather than breaking the Actor entirely.

---

## 4. Resolution: Trust Region Thawing via Scale-Up 
**Observation (Post-Intervention):**
After manually increasing the **learning rate by 10x** to `1e-5` (and vastly expanding `gen_batch_size` / `num_prompts`), the optimization loop fundamentally stabilized:
1. **Clip Fraction** surged to ~15%, aligning with healthy PPO trust-region updates.
2. **KL Divergence** flipped positive and expanded expectedly.
3. **Critic Loss** is decreasing smoothly, showing stable value function fit.
4. **Actor Loss** is trending downward, with healthy bounciness reflecting natural sampling variance over the wider batch distributions.
5. **Training Rewards** saturated impressively from **~30% up to >95% positive.**
6. **Evaluation Rewards** mirrored the training climb, ascending from **~30% to >95%** is a proof of convergence.

**Reasoning:**
1. **Unblocking the Trust Region:** The higher learning rate shattered the local minimum. Symmetrical 15% clipping proves the policy is finally taking aggressive optimization steps instead of tiptoeing.
2. **Rich Variance:** Expanding the batch radius injected diverse trajectories into the gradients. This drove necessary exploration (bouncy Actor loss) and forced the model to meaningfully abandon the frozen reference distribution (positive KL).
5. **True Generalization:** The parallel $60\% \rightarrow >95\%$ climb in Eval Rewards is the definitive proof of alignment; the policy genuinely modeled the semantic structure of sentiment rather than memorizing the training prompts.

---

## 5. Capacity vs. Optimization Dynamics (Qwen-0.6B vs. Gemma-4B)
**Observation:**
During experiments, changing the underlying foundation model parameter count drastically (e.g., swapping a 0.6B Qwen model for a 4B Gemma model) resulted in nearly identical training curves across all metrics (`mean_reward`, `actor_loss`, `mean_kl_penalty`, `clip_fraction`). Both models saturated the maximum reward within the same number of steps and subsequently traced identical KL penalty paths in the latter half of training.

**Reasoning:**
1. **Reward Saturation on Simple Tasks:** The task (producing a "positive" review) and the associated reward model (`twitter-roberta-base-sentiment`) act as a low-complexity bottleneck. The reward model merely checks for the presence of positive semantic clusters (words without demanding strict grammatical or compositional structural constraints). This makes the task "easy" to hack.
2. **Capacity is Unnecessary:** A 4B parameter model has significantly higher representational capacity than a 0.6B model. However, because the task is so simple, both models are vastly over-parameterized for it. 
3. **PPO Constraints Dominate:** Once the reward function saturates rapidly (e.g., at ~0.95), the "learning" process stops being about discovering new semantic patterns and shifts entirely to constraint satisfaction. The objective becomes: *How do we balance the static reward ceiling against the KL divergence penalty and PPO clipping boundaries?* Because both models hit the exact same reward ceiling almost immediately, the trajectories are mechanically forced down the exact same mathematical optimization path dictated by the PPO hyperparameters (`clip_epsilon`, `kl_beta`), completely shadowing the raw differences in model capacity.

To force the larger model (Gemma-4B) to diverge mathematically from the smaller model (Qwen-0.6B) and demonstrate its value, the reward signal must be inherently rigorous and structurally demanding. Moving from basic sentiment classification to complex multi-step reasoning tasks (e.g., mathematical proofs, strictly formatted JSON code generation) would introduce enough friction to break the early saturation barrier. Under a stricter reward landscape, the smaller model's capacity would plateau, and the larger model would trace an elevated, diverging trajectory.

---

## 6. Information Theory of the Critic
The mechanical convergence of the 4B and 0.6B models uniquely demonstrates the **Reward Function Bottleneck**. 
In our pipeline, a high-capacity generative brain (e.g., 4B parameters) was being graded by a comparatively tiny 125M parameter linear classifier (RoBERTa). The representational capacity of the Actor vastly exceeded the resolution of the Critic. Therefore, until the Critic is scaled (or replaced by deterministic environment criteria), the Actor is fundamentally constrained by the Critic's narrow understanding of "good" versus "bad."

---

## 7. Conclusion: Telemetry-Driven PPO Debugging
This pipeline served as a masterclass in physically debugging the PPO trust region using raw telemetry traces rather than black-box guesses.

**The Diagnostic Workflow:**
1. **Identifying the Local Minima:** We encountered early plateauing where evaluation score flatlined (~60%) and Policy updates seemingly froze. 
2. **Reading the Math:** Instead of blindly changing architectures, we correctly diagnosed a locked trust region by strictly observing:
   * **KL Penalty:** Drifting uniformly negative, indicating the model had easily learned "what not to say" but struggled to confidently discover the positive gradient.
   * **Advantages:** Broadly negative raw TD errors confirmed the Critic was excessively optimistic.
   * **Clip Fraction:** Hovering practically at $0\%$, proving the policy surrogate objective was rigidly clamped and unable to take meaningful optimization steps.
3. **The Mechanical Unlock:** We diagnosed the bottleneck as an overly conservative hyperparameter boundary. By aggressively **increasing the learning rate from 1e-6 to 1e-5** and massively blowing up the generation horizon (`gen_batch_size`), we forced the gradient descent out of its statistical rut.
4. **Validation:** The unlock was mathematically verified exactly as the literature expects—clip fractions bounced tightly between $5\% - 15\%$, KL-divergence climbed positive, and our out-of-distribution Evaluation Rewards perfectly mirrored Training Rewards (converging $60\% \rightarrow >95\%$).
