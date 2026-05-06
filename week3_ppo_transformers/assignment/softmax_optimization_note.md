# Memory Optimization: Cross Entropy vs Log Softmax

In Reinforcement Learning for LLMs, calculating `log_softmax` over an entire vocabulary (e.g., 150,000 tokens) creates a massive intermediate float tensor in VRAM. 

If you only want the log-probability of the *single* token you actually generated, allocating memory for the other 149,999 tokens is highly inefficient and leads to Out-Of-Memory (OOM) crashes on large batch sizes.

Instead, we can use a built-in PyTorch mathematical shortcut: **Negative Cross Entropy**.

### The Mathematical Shortcut

Recall the definition of Cross Entropy Loss for a target token $y$:

$$ \text{CE}(x, y) = -\log \left( \frac{\exp(x_y)}{\sum_i \exp(x_i)} \right) = - \text{log\_softmax}(x)[y] $$

Because PyTorch's `F.cross_entropy` is heavily optimized with fused CUDA kernels, it calculates the log-sum-exp denominator natively, but it **only** computes the numerator for your specific target $y$. 

Therefore, by simply multiplying the Cross Entropy by `-1`, you perfectly extract the exact log-probability tensor you need without materializing the rest of the vocabulary:

$$ - \text{CE}(x, y) = \text{log\_softmax}(x)[y] $$

It never allocates the massive `(Batch, Sequence Length, Vocabulary Size)` tensor in GPU memory, saving you Gigabytes of VRAM.

### Implementation

You can replace your explicit `log_softmax` and `gather` code with:

```python
import torch.nn.functional as F

# labels shape needs to be (B, T-1) for cross entropy
labels = generation_ids[:, 1:]

# F.cross_entropy expects classes/vocab to be in the second dimension: (B, V, T-1)
logits_transposed = logits.transpose(1, 2)

# Compute fused -log(P(target))
log_probs = -F.cross_entropy(logits_transposed, labels, reduction="none") # -> Output: (B, T-1)
```
