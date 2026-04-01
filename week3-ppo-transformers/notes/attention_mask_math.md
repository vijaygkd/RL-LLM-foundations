# Attention Masks in Causal LMs

## 1. Scaled Dot-Product Attention (base form)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $Q \in \mathbb{R}^{T \times d_k}$ — query matrix
- $K \in \mathbb{R}^{T \times d_k}$ — key matrix  
- $V \in \mathbb{R}^{T \times d_v}$ — value matrix
- $T$ — sequence length, $d_k$ — head dimension

The raw attention score matrix $A = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$, where entry $A_{ij}$ measures how much token $i$ attends to token $j$.

---

## 2. Adding a Mask

A mask matrix $M \in \mathbb{R}^{T \times T}$ is **added** to the raw scores before softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Where:
- $M_{ij} = 0$ → allow attention from token $i$ to token $j$
- $M_{ij} = -\infty$ → block attention (softmax maps $-\infty \to 0$)

---

## 3. The Causal Mask

Prevents any token from attending to future tokens (upper triangle is $-\infty$):

$$M^{\text{causal}}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

For a sequence of length 5:

$$M^{\text{causal}} = \begin{pmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

---

## 4. The Padding Mask

Given `attention_mask` $a \in \{0, 1\}^T$ from HuggingFace (1 = real token, 0 = pad):

$$M^{\text{pad}}_{ij} = \begin{cases} 0 & \text{if } a_j = 1 \quad (\text{real token}) \\ -\infty & \text{if } a_j = 0 \quad (\text{pad token}) \end{cases}$$

This blocks out the **entire column** for pad tokens — every token is blocked from attending to that position.

---

## 5. Combined Mask (what the model actually uses)

$$M^{\text{total}} = M^{\text{causal}} + M^{\text{pad}}$$

A position is attended to only if it is (a) in the past AND (b) a real token.

---

## 6. Why this matters: Left-Padding

HuggingFace left-pads batches for causal LM generation so the last real token is aligned across all sequences in the batch.

Example: prompt = `[PAD] [PAD] [The] [movie] [was]`  
`attention_mask = [0, 0, 1, 1, 1]`

**Without padding mask** (causal only), token `The` (position 2) can attend back to both PAD positions (0, 1):

$$\text{output}[\text{The}] = \alpha_0 \cdot V[\text{PAD}] + \alpha_1 \cdot V[\text{PAD}] + \alpha_2 \cdot V[\text{The}]$$

Since $V[\text{PAD}]$ is a **real learned embedding vector** (not zeros), this corrupts the representation. The model was always trained with pad tokens masked, so this is out-of-distribution.

**With padding mask**, columns 0 and 1 are fully blocked → $\alpha_0 = \alpha_1 = 0$ → `The` only attends to itself. Correct behavior restored.

---

## 7. Why Right-Padding is Safe for Causal LMs

If PADs are at the end: `[The] [movie] [was] [PAD] [PAD]`

The causal mask already blocks `movie` from seeing `PAD` tokens (they're in the future). So right-padding doesn't contaminate real tokens — **but HuggingFace still uses left-padding for generation** because it needs all sequences in a batch to be ready to generate the next token at the same position.

---

## 8. Practical Implication for PPO

When computing `get_log_probs_and_values`:

```python
# Correct attention mask for generation_ids (left-padded prompts + generated tokens)
gen_attn_mask = torch.cat([
    input_attn_masks,                                            # (B, T_prompt) — may have 0s from left-padding
    torch.ones(B, T_new, device=generation_ids.device)          # (B, T_new)    — always 1 (no padding here)
], dim=1)   # (B, T_total)
```

If your IMDB prompts are all truncated to exactly `prompt_token_len=8` with no padding, `input_attn_masks` is all 1s and this is a no-op — but it becomes critical once you use variable-length prompts.

---

## 9. How Masking Affects Backpropagation

**Short answer:** Yes — the zero softmax weight from a masked position does block gradients through that path. But the mechanism is more subtle than just "weight = 0 → no gradient."

### The Softmax Jacobian

For a softmax output $\alpha = \text{softmax}(z)$, the Jacobian is:

$$\frac{\partial \alpha_k}{\partial z_j} = \alpha_k (\delta_{kj} - \alpha_j)$$

For a **masked position** $j$ where $z_j = -\infty$, we have $\alpha_j = 0$. Plugging in:

$$\frac{\partial \alpha_k}{\partial z_j} = \alpha_k (0 - 0) = 0 \quad \text{for all } k$$

So the gradient of any downstream loss $L$ with respect to the raw score $z_j$ of a masked position is:

$$\frac{\partial L}{\partial z_j} = \sum_k \frac{\partial L}{\partial \alpha_k} \cdot \frac{\partial \alpha_k}{\partial z_j} = 0$$

Since $z_{ij} = Q_i K_j^T / \sqrt{d_k}$, both the query gradient (token $i$ trying to attend to $j$) and the key gradient (token $j$ being attended to) are **zero through this path**.

### What this means for pad tokens

With the padding mask correctly applied:
- No real token can attend to a PAD token → PAD key/value vectors receive no gradient through attention weights ✓
- PAD output positions do not contribute to the loss (loss mask) → no gradient flows backward from the loss through PAD output positions ✓

Combined, the PAD token embedding accumulates **effectively zero gradient** when masked correctly.

### Caveat 1: Residual gradient paths in left-padded sequences

With left-padding `[PAD_0] [PAD_1] [The] ...`, PAD_0 can still attend to nothing (causal mask blocks it from attending forward, and there's nothing before it). PAD_1 can attend back to PAD_0. These PAD-to-PAD attention paths do pass gradients — but since the loss is also zeroed at PAD output positions, they don't produce meaningful gradient signal.

### Caveat 2: The NaN trap

If the **entire row** of a token is masked (every position in its past is a pad token), then the softmax input is all $-\infty$:

$$\text{softmax}(-\infty, -\infty, \ldots) = \frac{0}{0} = \text{NaN}$$

This can happen with an overly aggressive padding mask (e.g., the first PAD token in a fully left-padded batch that can only attend to other PAD tokens, which are also masked). PyTorch does **not** guard against this automatically. The fix is to ensure at least one unmasked position per row — which is why some implementations set $M^{\text{pad}}_{ii} = 0$ for all $i$ (a token can always attend to itself).

### Summary table

| Scenario | Gradient blocked? |
|---|---|
| Real token $i$ attending to masked position $j$ | ✅ Yes — zero softmax weight → zero gradient through this path |
| PAD position's own output contributing to loss | ✅ Yes — loss mask zeros it out |
| PAD-to-PAD attention in left-padded sequences | ⚠️ Gradient flows between PADs, but loss mask prevents it from mattering |
| All positions in a row masked | ❌ NaN — must ensure at least one unmasked position per row |
