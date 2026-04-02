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
