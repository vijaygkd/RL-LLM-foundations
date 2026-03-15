## PPO 

### The One-Liner:
* The paper proposes a new objective function (clipped surrogate objective) for policy-gradient RL that alleviates the instability of previous methods by clipping the probability ratios to prevent large updates to the policy.


### The Math (Clipped Surrogate Loss Function): 

Use log-prob for ratio to avoid NaN

$r_t = torch.exp(log(p_new) - log(p_old))$

$\mathcal{L} = -(min(r_t.A, clip(r_t, 1-\epsilon, 1+\epsilon).A)).mean()$


### Hyperparameters: 
* $\epsilon = 0.2$ (empirical)
* Discounted Return of Value model $\gamma = 0.99$
* GAE parameter $\lambda = 0.95$
* Adam learning rate = 3e-4
* Batch size = 64
* Horizon T = 512-2048



### Gotchas: 
* When $A > 0$ and $r > 1 + \epsilon$, the objective function becomes $\mathcal{L} = (1 + \epsilon) . A$. Thus gradient of $\mathcal{L}$ wrt policy $\theta = 0$. The allows PPO to "ignore" any sample updates that can cause large updates to policy.
* **Intuitively:** PPO is "pessimistic." It ignores large "optimistic" steps (large gains) but allows large "pessimistic" steps (correcting errors). This asymmetry leads to smoother, more stable convergence.
* Improvements are seen across many different tasks
* PPO algorithm runs multiple rollouts (N) for several timestamps (T). Then runs mini-batch SGD on the NT updates for each sample by summing.
* Value model is usually a seperate model trained sometimes with shared frozen weights.


----------

## Generalized Advantage Estimation (GAE) & The Critic

**Goal:** Understand how we calculate "Advantage" ($A_t$) to solve the Credit Assignment problem in RL.

## 1. The Cast of Characters
* **The Actor ($\pi_\theta$):** Decides *what to do* (Action).
* **The Critic ($V_\phi$):** Predicts *how good it is to be here* (Value).
    * $V(s_t)$ estimates the **sum of all future discounted rewards** from state $s_t$.
    * *Analogy:* If you are in a safe spot with full health, $V \approx 100$. If you are standing on a trap, $V \approx 10$.

## 2. The Intuition: Why not use Raw Rewards?
If we only use raw rewards (e.g., "I won, +1"), we have high **Variance**.
* *Scenario:* You make a blunder at step 50 but get lucky at step 99 and win.
* *Raw Reward:* The model thinks the blunder was genius because the final result was +1.
* *Advantage:* We measure how much **better or worse** an action was compared to the average expectation.

$$\text{Advantage} = \text{Actual Outcome} - \text{Expected Outcome (Baseline)}$$

* If you won (+1) but the Critic *expected* you to win ($V=0.9$), the Advantage is small ($0.1$). You just did your job.
* If you won (+1) but the Critic thought you were doomed ($V=0.1$), the Advantage is huge ($0.9$). Brilliant move!

## 3. The Math: The TD Error ($\delta_t$)
The term $\delta_t$ represents the **"One-Step Surprise."**

$$\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{Reality}} - \underbrace{V(s_t)}_{\text{Expectation}}$$

* **Reality:** Reward received ($r_t$) + Discounted value of the next state ($\gamma V(s_{t+1})$).
* **Expectation:** The Critic's original prediction for the current state ($V(s_t)$).
* **Interpretation:**
    * $\delta_t > 0$: Reality was better than expected (Good surprise).
    * $\delta_t < 0$: Reality was worse than expected (Bad surprise).

## 4. The Magic: Generalized Advantage Estimation (GAE)
$\delta_t$ only looks one step ahead. GAE allows us to look $N$ steps ahead by summing discounted future surprises.

$$A_t = \delta_t + (\gamma \lambda) A_{t+1}$$

This is a recursive formula (weighted sum into the future).
* **If $\lambda = 0$:** $A_t = \delta_t$. Only immediate surprise. (Low variance, high bias).
* **If $\lambda = 1$:** $A_t = \sum \delta$. Sum of all future surprises. (High variance, low bias).
* **GAE ($\lambda \approx 0.95$):** The sweet spot. It uses the immediate surprise + a discounted chunk of the next surprise.

## 5. Implementation Note
When calculating GAE in code, you iterate **backwards** from the last step of the rollout to the first.

```python
# Pseudo-code for GAE loop
gae = 0
for step in reversed(range(num_steps)):
    # Calculate delta
    delta = rewards[step] + gamma * V(next_state) * (1-done) - V(current_state)
    
    # Recursive update
    gae = delta + gamma * lambda * gae
    
    advantages[step] = gae
```
