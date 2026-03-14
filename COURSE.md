### **Course Title: Reinforcement Learning for LLM Post-Training**

**Goal:** Master the techniques used to align LLMs, from PPO foundations to modern Direct Preference Optimization (DPO) and reasoning-focused RL.
**Duration:** 12 Weeks (3 Months)
**Prerequisites:** Knowledge of LLM Architecture (Transformers), Backprop, SFT, Python/PyTorch.

---

### **Month 1: The Foundations (PPO & The RLHF Loop)**

*Objective: Understand the "Classic" RLHF pipeline used by InstructGPT/ChatGPT and implement the PPO algorithm.*

#### **Week 1: RL Primitives & The Language Modeling Problem**

* **Concept:** Mapping RL concepts to LLMs.
  * Agent = The LLM.
  * State = The prompt + tokens generated so far.
  * Action = The next token selected from the vocabulary.
  * Reward = A scalar score for the completed sequence (from a Reward Model).

* **Reading:**
  * *Proximal Policy Optimization Algorithms* (Schulman et al., 2017) - The holy grail paper.
  * *Deep Reinforcement Learning from Human Preferences* (Christiano et al., 2017).

* **Assignment:** "Gridworld PPO." Implement PPO from scratch in PyTorch for a simple Gymnasium environment to understand the loss function (Clipped Surrogate Objective) before applying it to text.

#### **Week 2: The Reward Model (RM)**

* **Concept:** RL needs a signal. How do we train a model to mimic human ranking?
  * Bradley-Terry Model & Pairwise ranking loss.
  * Reward hacking/overoptimization.

* **Reading:**
  * *Training language models to follow instructions with human feedback* (InstructGPT - Ouyang et al., 2022).
  * *Scaling Laws for Reward Model Overoptimization* (Gao et al., 2022).

* **Assignment:** Train a Reward Model (BERT or tiny Llama) on the `Anthropic/hh-rlhf` dataset to classify which of two responses is better.

#### **Week 3: PPO for Transformers (The "TRL" Loop)**

* **Concept:** The KL-Divergence penalty (keeping the model close to the SFT reference). Value Heads. Generalized Advantage Estimation (GAE).
* **Reading:**
  * Hugging Face `TRL` (Transformer Reinforcement Learning) documentation.

* **Assignment:** "Sentiment Steering." Use PPO to fine-tune a small GPT-2 model to generate *only* positive movie reviews, using a sentiment classifier as the Reward Model.

#### **Week 4: Implementation Challenge**

* **Project:** Setup the full RLHF loop.
  1. Start with an SFT model (e.g., Llama-3-8B-Instruct).
  2. Load a Reward Model.
  3. Run PPO to optimize the SFT model against the RM.

* *Goal:* Observe the "alignment tax" (does the model lose creativity?) and tune the KL coefficient.

---

### **Month 2: The Modern Era (DPO & Preference Optimization)**

*Objective: Move away from complex PPO pipelines to "Contrastive" methods that are stable and efficient.*

#### **Week 5: Direct Preference Optimization (DPO)**

* **Concept:** Deriving the optimal policy analytically without a separate Reward Model. The "implicit" reward formulation.
* **Reading:**
  * *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023).

* **Assignment:** Implement the DPO Loss function in PyTorch (it's just a few lines of code!) and compare it to the PPO objective mathematically.

#### **Week 6: DPO Variations (IPO, KTO, ORPO)**

* **Concept:** Solving DPO's instability.
  * **IPO (Identity Preference Optimization):** Adding regularization to prevent overfitting.
  * **KTO (Kahneman-Tversky Optimization):** Learning from binary good/bad labels instead of pairs.

* **Reading:**
  * *KTO: Model Alignment as Prospect Theoretic Optimization* (Ethayarajh et al., 2024).

* **Assignment:** "Dataset Engineering." Create a preference dataset (chosen vs. rejected) and fine-tune a model using KTO.

#### **Week 7: Iterative & Online DPO**

* **Concept:** Static DPO has limits. Online DPO involves the model generating its *own* data during training, which is then judged by an external oracle or RM.
* **Reading:**
  * *Self-Rewarding Language Models* (Yuan et al., 2024).

* **Assignment:** Simulate an "Online" loop: Generate 100 responses with your current model, score them with an RM, construct a new DPO dataset, and train for one epoch. Repeat.

#### **Week 8: Month 2 Capstone**

* **Project:** "The Aligned Assistant."
  * Take a strong base model (e.g., Mistral or Llama).
  * Apply DPO using the `UltraFeedback` dataset.
  * Evaluate using `MT-Bench` or `AlpacaEval` to see if your win-rate improved over the SFT baseline.

---

### **Month 3: Advanced Frontiers (Reasoning & Speculative RL)**

*Objective: Explore the cutting edge—RL for "System 2" thinking (Chain of Thought) and Process Supervision.*

#### **Week 9: Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)**

* **Concept:** Rewarding the *steps* of reasoning (e.g., in Math) rather than just the final answer.
* **Reading:**
  * *Let's Verify Step by Step* (Lightman et al., 2023 - OpenAI).

* **Assignment:** Math verifier. Train a small classifier to verify steps in a GSM8K math solution.

#### **Week 10: Search & Planning (STaR and Tree Search)**

* **Concept:** Reinforcement Learning as inference-time search.
  * **STaR (Self-Taught Reasoner):** Bootstrapping reasoning by filtering for correct answers.

* **Reading:**
  * *STaR: Bootstrapping Reasoning With Reasoning* (Zelikman et al., 2022).

#### **Week 11: Group Relative Policy Optimization (GRPO)**

* **Concept:** The technique behind models like DeepSeek-R1. Optimizing groups of outputs relative to each other without a value function critic, reducing memory overhead.
* **Reading:**
  * *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (DeepSeek AI, 2024).

#### **Week 12: Final Capstone Project**

* **Goal:** Build a "Reasoning Specialist."
  * Fine-tune a model to solve complex logic puzzles.
* **Method:**
  1. SFT on Chain-of-Thought data.
  2. Generate multiple candidate solutions per prompt.
  3. Use a simple verifier (e.g., checking if code compiles or if the answer matches the ground truth) to label them.
  4. Apply DPO/GRPO to prefer the correct reasoning paths.
