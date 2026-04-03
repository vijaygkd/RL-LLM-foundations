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
5. **Training Rewards** saturated impressively from ~30% up to >95% positive.
6. **Evaluation Rewards** mirrored the training climb, ascending from ~60% to >95%.

**Reasoning:**
The previous $10^{-6}$ learning rate artificially choked the Actor into a local minimum. By releasing that constraint and scaling the batch radius, we supplied a rich variance of trajectories. The Actor properly pushed against the $1 \pm 0.2$ bounds, aggressively updating parameters for the good trajectories. The "bounciness" in Actor loss is a positive signal—it proves the algorithm is exploring diverse semantic tokens rather than clinging safely to the reference model. Most importantly, the symmetric rise in Evaluation Rewards definitively proves that the model is **learning generalizable sentiment features** rather than simply memorizing the training subset or hacking a sequence length loophole.
