# KL Divergence Penalty in PPO

## Your Observation (Correct!)

The per-token penalty is computed as the log-ratio:

$$\text{penalty}_t = \log \pi_\theta(a_t) - \log \pi_{ref}(a_t)$$

You correctly identified three cases:

| Case | Condition                | Log-ratio | Effect on reward                   |
| ---- | ------------------------ | --------- | ---------------------------------- |
| 1    | $\pi_\theta = \pi_{ref}$ | $= 0$     | No penalty                         |
| 2    | $\pi_\theta > \pi_{ref}$ | $> 0$     | Penalized (subtracted)             |
| 3    | $\pi_\theta < \pi_{ref}$ | $< 0$     | **Negative penalty = small bonus** |

Case 3 is what you found counterintuitive — if the model assigns *less* probability to a token than the reference, the penalty is negative, which slightly *helps* the reward. Are we encouraging divergence?

---

## Why This is Not a Problem

### 1. The per-token log-ratio is a Monte Carlo estimator, not the true KL

The **true KL divergence** is defined as the *expectation* of the log-ratio:

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{a \sim \pi_\theta}\left[\log \frac{\pi_\theta(a)}{\pi_{ref}(a)}\right]$$

This is **always ≥ 0** (by Gibbs' inequality). It equals 0 *only* when $\pi_\theta = \pi_{ref}$ everywhere.

The per-token calculation is just **one sample** from this expectation. A single sample *can* be negative — that is the variance of the estimator, not a logical flaw. Over many tokens and rollouts the average converges to the true (non-negative) KL.

True KL requires: $$= \sum_{a \in V} \pi_\theta(a) \cdot \left( \log \pi_\theta(a) - \log \pi_{ref}(a) \right)$$ 

### 2. Probability distributions must sum to 1

This is the key constraint that prevents exploitation of Case 3.

The policy $\pi_\theta$ is a probability distribution over the full vocabulary (e.g., 50,000 tokens). **It must always sum to 1.**

If the policy decreases its probability mass on *some* tokens below the reference, it is **forced by the sum-to-1 constraint** to increase it on *other* tokens. Those tokens where $\pi_\theta > \pi_{ref}$ will incur a positive penalty that more than compensates for the negative penalty elsewhere.

The policy cannot "win" by globally shifting below the reference — it's mathematically impossible for all tokens simultaneously.

---

## Summary

A single negative per-token KL penalty is just noise in a Monte Carlo estimate. In expectation, across the full vocabulary distribution and across many rollouts, the aggregate KL penalty is always non-negative and correctly penalizes *any* distributional shift away from the reference — whether the model becomes more confident or less confident on any given token.
