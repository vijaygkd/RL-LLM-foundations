## DRL-HF (Deep Reinforcement Learning from Human Feedback)

### The One-Liner:
* Learn a reward model using human preference labels comparing pairs of trajectory segments, then train the policy with RL instead of using per-step environment rewards.

### The Math:
Train $r$ using binary cross-entropy loss.

$$
\mathcal{L}(r) = -\sum \big(y \cdot P(x_1 > x_0) + (1-y) \cdot P(x_0 > x_1)\big)
$$

### Hyperparameters (TRPO):
* discount rate $\gamma = 0.95$
* $\lambda = 0.97$
* entropy bonus $= 0.01$

### Gotchas:
* Real-world environments don't provide a continuous reward per step.
* Human-preference feedback comparing pairs of trajectories is simple to obtain.
* A comparatively small amount of comparison data is required (e.g., a few hours of comparisons).
* Math: the difference in predicted rewards between two trajectories corresponds to the probability of one segment being chosen over another.
* Data: stored as triplets `(seg1, seg2, win{1,2})`.
