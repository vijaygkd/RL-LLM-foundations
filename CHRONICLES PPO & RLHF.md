# Deep Learning Chronicles: Implementation Gotchas & Reality

This document serves as a living repository of counter-intuitive bugs, theoretical discrepancies, and engineering realities encountered running modern deep learning and RLHF algorithms from scratch. 

## Week 3 - Reinforcement Learning from Human Feedback (RLHF) / PPO

### 1. The PPO Shifting Target Bug (Lazy Evaluation)
*   **The Gotcha:** It is tempting to write PPO loops that lazily evaluate `old_log_probs` and `Advantages` micro-batch by micro-batch to save memory. 
*   **The Reality:** If you evaluate these inside the gradient update loop, the actor weights change after the first batch. By batch 2, your $\pi_{old}$ is no longer the behavioral policy that gathered the data; it is the newly updated model. The PPO clipped surrogate objective $\frac{\pi_{new}}{\pi_{old}}$ mathematically requires $\pi_{old}$ to be absolutely static across the dataset.
*   **The Fix:** Run a pure `with torch.no_grad():` pre-computation loop over the entire generated rollout dataset *before* the learning epochs begin. Concatenate and freeze the `old_log_probs`, `ref_log_probs`, and `advantages` tensors.

### 2. Advantages as Static Environment Anchors
*   **The Theory:** In PPO, Advantages must be conceptually treated as a strictly static component of the data-generation environment. They are computed using the *old static policy* as an anchor to baseline all future learning updates.
*   **The Dual Purpose:** The static Advantage serves two separate update mechanics:
    1.  **Actor Policy:** It acts as a scalar weight that scales the $\frac{\pi_{new}}{\pi_{old}}$ probability ratio. The gradients flow exclusively through the log-probabilities to update the generative policy networks. 
    2.  **Value Critic:** It generates the static return target ($Target = V_{old} + \hat{A}_{old}$) that the critic dynamically regresses against.
*   **The Gotcha:** If you recompute the advantages using the active on-policy during the learning loop, you unmoor the mathematical anchor. Your learning targets continuously shrink and shift, destabilizing the Value MSE and allowing Critic gradients to chaotically bleed back into the Actor's log-probability optimization path.

### 3. Terminal State Value Leakage in GAE
*   **The Gotcha:** When calculating TD Errors ($\delta_t = r_t + \gamma V_{t+1} - V_t$), the neural network Critic outputs float values for all tokens, including `<EOS>` and `<PAD>` tokens.
*   **The Reality:** Mathematically, a terminal state has no future, so $V_{terminal}$ is strictly $0.0$. If you add $\gamma V(\text{EOS})$ to your TD error, your Generalized Advantage Estimation (GAE) will absorb random noise from the untrained PAD-token embeddings. 
*   **The Fix:** You must explicitly multiply the strictly right-shifted $V_{t+1}$ tensor by your boolean `<PAD>` mask *before* the TD error addition. 

### 4. Log-Softmax VRAM Hemorrhaging
*   **The Gotcha:** You want the log-probability of a generated sequence. The logical step is `log_probs = log_softmax(logits).gather(labels)`.
*   **The Reality:** Materializing `(Batch, SeqLen, Vocab_Size)` into VRAM for a 150,000+ token vocabulary consumes gigabytes of memory just to throw away 99.9% of the tensor via `gather()`. 
*   **The Fix:** Use `log_probs = -F.cross_entropy(logits_transposed, labels, reduction="none")`. PyTorch's fused CUDA kernels compute the mathematical equivalent of `log_softmax(x)[y]` on the fly, skipping the massive intermediate matrix allocation.

### 5. The Action Mapping (Off-by-One Anxiety)
*   **The Gotcha:** Masking sequences for advantage computation feels confusing because `log_probs[t]` and `advantages[t]` correspond to the mask of token `t+1`.
*   **The Reality:** In Auto-Regressive Language Modeling, the RL environment "action" taken at step $t$ predicts the language token instantiated at step $t+1$. Left-shifting your Action/Log-prob matrices perfectly encapsulates this. Always evaluate the validity of Action $t$ against the presence of `<PAD>` at Token $t+1$. 

### 6. The Variable-Length Tensor Mapping Nightmare
*   **The Gotcha:** Mapping arbitrary natural language sequences to tensor indices for RL equations isn't straightforward due to padding constraints. 
*   **The Reality:** The generation phase intuitively requires **left-padding** (to align the current state for the autoregressive forwards pass), while the RL trajectory phase requires **right-padding** (because each rollout sequence concludes at a different terminal length).
*   **The Dual Index Problem:** We must simultaneously track three distinct tensor shapes:
    1. Full Sequence Tensors (Length $T$): `generation_ids`, padding `masks`, and `critic_values` (which extend all the way to $T$ to allow for terminal state bootstrapping).
    2. Left-Shifted Action Tensors (Length $T-1$): `log_probs`, `td_errors`, and `advantages`, which map sequentially to the *actions* taken to reach the next state.
*   **The Fix:** You must create an explicit `gen_output_mask` (Boolean: 1 for generated tokens, 0 for prompt or padding) and apply it relentlessly across your `T-1` action sequences to zero out RL gradient calculations over prompts and padded voids. Only timestamps where the policy actively generated a valid token receive gradient updates.

### 7. The Dual-Nature of Advantage Normalization
*   **The Gotcha:** You intuitively know you should normalize Advantages ($\mu=0, \sigma=1$) to stabilize training computations. So you instantly normalize the Advantage matrix and use it everywhere.
*   **The Reality:** The Advantage matrix mathematically serves two completely distinct purposes in the PPO update loop that require entirely different scaling treatments.
*   **The Dichotomy:**
    1.  **Actor Policy (Requires Normalization):** For the Actor gradient update, the Advantage acts as a relative scalar weight determining how heavily to maximize the probability ratio. Normalizing it binds the statistical variance of your gradient steps, stabilizing your learning rate regardless of arbitrarily extreme reward distributions.
    2.  **Value Critic (Requires Raw Data):** For the Critic loss, you construct an absolute scalar Target Return ($V_{old} + \hat{A}_{raw}$). If you build this regression target using normalized advantages, you systematically warp the absolute metric magnitude your critic is evaluating, causing the network's internal values to hopelessly decay and drift.
*   **The Fix:** Compute your absolute Target Returns analytically using the raw Advantages. Only *after* computing the targets should you normalize the Advantages for the Actor to ingest.

### 8. The PPO Generation Strategy (Batches vs. Epochs & Prompt Diversity)
*   **The Gotcha:** You might intuitively want to generate rollouts over your entire dataset (e.g., 50,000 prompts) before running updates, or generate multiple rollouts per prompt.
*   **The Reality:** PPO thrives on rapid feedback and strict baseline comparisons. 
    1. **Rollout Batches:** Computing rollouts across the complete dataset delays the gradient loop. By the time the trainer analyzes the last sequence, it is updating against mathematically stale probabilities.
    2. **Rollouts per Prompt:** You strictly want 1 rollout per prompt in PPO. A separate Critic network establishes the value baseline. In algorithms like GRPO, 8-16 rollouts are required because it drops the Critic entirely and relies on the mean reward of those specific rollouts as its baseline.
*   **The Fix:** Configure your generator to pull moderately sized rollout chunks (e.g., 512 prompts), enforce exactly 1 rollout per prompt to maximize diversity, execute the learning epochs, and discard the data for fresh inferences.

### 9. Aggressive Vectorization in RL Environments
*   **The Gotcha:** A naive PPO implementation might use nested `for` loops to iterate over batches, sequences, and timestamps to calculate Rewards, KL penalties, TD Errors, and Advantages.
*   **The Reality:** Python control flow is slow. Looping over data on the CPU prevents the GPU from parallelizing matrix operations, turning mathematical calculation into the primary training bottleneck.
*   **Pro tip:** You must aggressively vectorize environment calculations:
    1. Base variables (like aligning scalar Rewards to the `<EOS>`) should be injected at specified indices via PyTorch tensor mapping methods like `scatter_()` (`scatter` is opposite of what `gather` does)
    2. TD-errors ($\delta_t = r_t + \gamma V_{t+1} - V_t$) must be calculated as a single monolithic matrix subtraction across the entire `(Batch, SeqLen)` grid instantly.
    3. The only acceptable `for` loop is the recursive Generalized Advantage Estimate (GAE) calculation, as $A_t$ inherently relies on $A_{t+1}$. Even then, that loop must only iterate over the sequence dimension `T`, remaining strictly vectorized across the massive batch axis `B`.

### 10. Reward Model Capacity Bottleneck (Reward Saturation)
*   **The Gotcha:** Training a 4B parameter model (Gemma) vs a 0.6B parameter model (Qwen) on the same dataset yields identical optimization curves and identical plateau points.
*   **The Reality:** A 125M parameter sentiment reward model is too simple to differentiate the capacities of a 0.6B and 4B model. Both actors achieve the maximum reward (~0.95) quickly. Once the reward function saturates, there is no further gradient signal for the larger model to utilize.
*   **The Rule:** To leverage larger parameter models, the reward landscape must be structurally demanding (e.g., compile-time unit tests for code, formal verifiers for math). A simple reward signal causes PPO optimization to collapse into identical clipping constraints regardless of underlying model size.

### 11. Diagnosing a Frozen PPO Trust Region
*   **The Gotcha:** The PPO training loop runs without errors, but evaluation scores flatline at baseline levels (~60%) and the policy does not update.
*   **The Reality:** Training is stuck in a local minimum if the telemetry shows:
    1.  **KL Divergence:** Drifting negative (the model lowers token probabilities to avoid penalties but fails to locate positive output paths).
    2.  **Advantages:** Predominantly negative (the Value Critic consistently overestimates returns).
    3.  **Clip Fraction:** Near 0.0% (the policy updates are too small to ever hit the $1 \pm 0.2$ PPO boundary).
*   **The Fix:** Increase the learning rate (e.g., `1e-6` to `1e-5`) and increase the generation batch size (e.g., `gen_batch_size=512`). This injects trajectory variance into the batches. A successful optimization step will register a 5% to 15% clip fraction and drive KL divergence positive.

## Week 2 - Reward Modeling & Preference Learning

### 1. The Capacity Bottleneck (Linear Probes vs. Fine-tuning)
*   **The Gotcha:** It feels intuitive and computationally efficient to freeze a large pretrained model's backbone and only train the final scalar reward head (a linear probe) on preference data.
*   **The Reality:** We found that a frozen 0.5B model with only a trainable `lm_head` barely crosses the 50% random baseline (plateauing around 51-53% accuracy) on the HH-RLHF dataset. The underlying semantic representations of a base neural network are not inherently aligned to human preference nuances.
*   **The Fix:** You must unfreeze the transformer backbone (at minimum the last 4-8 layers, ideally full-parameter fine-tuning if VRAM allows) so the network can actively reshape its deep representations to map subjective preference concepts. Full unfreezing instantly pushed our test accuracy from ~51% to 64.4%.

### 2. The "1-Epoch" RLHF Maxim
*   **The Gotcha:** You assume normal deep learning rules apply: if accuracy is solidly climbing at the end of epoch 1, you should train for 3-5 epochs until saturation.
*   **The Reality:** Human preference datasets are heavily contrived and highly brittle. The moment epoch 2 began, our loss curve instantly flattened into a horizontal pancake and accuracy ping-ponged randomly. The model abandoned learning generalized preference features and immediately began memorizing specific prompt-response strings.
*   **The Rule:** The 1-Epoch maximum. In RLHF reward modeling, train your model on the dataset exactly once, extract the maximum generalized signal, and stop.

### 3. VRAM Tetris (Hardware vs. State Memory)
*   **The Gotcha:** Pushing a `BATCH_SIZE` of 512 through an Nvidia H200 (141GB VRAM) for a tiny 0.5B parameter model seems trivial until you completely unfreeze the network.
*   **The Reality:** While the 0.5B weights conceptually only take ~1GB in `bfloat16`, full unfreezing triggers a catastrophic explosion in required activation memory. `torch.compile` caching, Adam optimizer momentum buffers, and storing the forward-pass activations for a `512` batch size at a `768` sequence length causes instant Out-of-Memory (OOM) failures even on 141GB of VRAM.
*   **The Fix:** Drop the literal forward-pass tensor batch size to highly defensive levels (e.g., `BATCH_SIZE = 64`) and utilize `GRAD_ACCUM_STEPS = 8`. This perfectly emulates the optimization landscape of a 512 batch while protecting the memory bandwidth overhead.

### 4. Surgical Reward Extraction (The Padding Gotcha)
*   **The Gotcha:** Passing an autoregressive batch through a scalar output head yields a tensor of shape `(Batch, SeqLen, 1)`. You intuitively want the final element: `logits[:, -1, :]`.
*   **The Reality:** Due to dataset collation and right padding, `logits[:, -1, :]` evaluates the scalar reward of a `<PAD>` token, yielding pure mathematical noise for the Bradley-Terry ranking loss.
*   **The Fix:** You must dynamically track the exact integer sequence length of each prompt-response pair in the batch before padding is applied. You then use `torch.gather()` on the sequence axis to surgically extract the discrete float from the *last non-padded* generative token.

### 5. Architectural Cleanliness (The "God-Loop" Trap)
*   **The Gotcha:** Novice researchers often pile logging, matplotlib plotting, metric smoothing, and wandb syncing directly into the core `for epoch` loop, turning the mathematical heart of the algorithm into an unreadable 300-line script.
*   **The Reality:** The cognitive load required to debug tensor shapes and RL loss functions is extremely high. Visual clutter kills your ability to spot algorithmic flaws.
*   **The Fix:** Strictly isolate all non-mathematical logic (Telemetry, Checkpointing, Data Visualization) into a separate class (e.g., `TrainingTelemetry` in `utils.py`). The core training loop should read purely like the mathematical pseudocode presented in the academic paper.

### 6. Data-Driven Truncation over Blind Assumptions
*   **The Gotcha:** A standard base model like Qwen/Llama supports 4096+ tokens. It is tempting to set `max_length=1024` or `2048` "just to be safe" when preparing your dataloader to prevent clipping.
*   **The Reality:** Sequence length dictates activation memory quadratically (in attention matrices) and linearly everywhere else. Wasting padding space is devastating for batch size throughput. 
*   **The Fix:** Never blindly guess sequence limits. We wrote a dedicated analysis utility to map the token distribution of our specific `Anthropic/hh-rlhf` dataset and found that `max_length = 768` covered perfectly ~98.6% of all sequences. This exact calculation allowed us to aggressively dial up the batch size without clipping vital preference data.

### 7. The Mathematics of Gradient Accumulation
*   **The Gotcha:** Treating gradient accumulation simply as "call backwards N times, then step" without addressing the loss scales. 
*   **The Reality:** By default, CrossEntropy and normal losses average over the mini-batch. If you accumulate 8 mini-batches without scaling the gradients down, you aren't simulating a large 512 batch; you are simulating a large batch *while also multiplying the effective learning rate by 8*, leading to explosive instability and instant NaN losses.
*   **The Fix:** You must explicitly divide your aggregated loss by `GRAD_ACCUM_STEPS` (i.e., `scaled_loss = loss / GRAD_ACCUM_STEPS`) before calling `.backward()`. This ensures the magnitude of the final accumulated gradient vector perfectly matches the magnitude of a single massive forward pass.
