# Week 2: The Reward Model (RM)

**Goal:** Train a "Judge" model that can read two responses and decide which one is better.

In CartPole, the environment gave us the reward (`+1` for every frame).
In the real world, the "Environment" is a human user. Humans are expensive and slow. We cannot ask a human to rate every single sentence our LLM generates during training.

**The Solution:** We train a **Reward Model (RM)** to mimic human preferences.
* **Input:** `(Prompt, Response)`
* **Output:** `Scalar Score` (e.g., 4.5/5.0)

We will then use this RM to replace the CartPole `+1` signal in Week 3.

### 1. The Primer: "Preference Learning"
We don't train RMs using Mean Squared Error (MSE) like a regression problem (e.g., "This text is a 7.2"). Humans are bad at absolute scoring.
* *Bad Question:* "Rate this summary from 1 to 10." (Subjective, noisy).
* **Good Question:** "Which summary is better: A or B?" (Comparative, stable).

We use the **Bradley-Terry Model** loss function. We feed the model a "Chosen" response ($y_w$) and a "Rejected" response ($y_l$) and minimize:
$$\mathcal{L}_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l))$$
* We want the scalar reward for "Winner" ($y_w$) to be higher than "Loser" ($y_l$).

### 2. Week 2 Reading List
Focus on **how** they construct the dataset and the specific loss function used for the Reward Model.

1.  **[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)**
    * *Focus:* **Section 3.2 (Reward Model)** and **Equation 2**.
    * *Key Insight:* They initialized the RM from the SFT model, not a random model. Why?
2.  **[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)**
    * *Focus:* **Section 3.1 (Reward Modeling)**.
    * *Key Insight:* They trained *two* Reward Models (one for Safety, one for Helpfulness) to avoid conflicting objectives.

### 3. Week 2 Coding Assignment (Preview)
* **Dataset:** We will use the `Anthropic/hh-rlhf` dataset (Helpful & Harmless), which contains pairs of `(chosen, rejected)` dialogues.
* **Model:** We will fine-tune a small transformer (like `DistilBERT` or `TinyLlama`) to predict which response is better.
* **Success Metric:** Your model should achieve **>65% accuracy** on the test set (predicting the human's choice).

### **Immediate Next Step**
Create `week2/` in your repo. Read the **InstructGPT** paper (Section 3.2).

**Discussion Question (The Gatekeeper):**
> In the InstructGPT paper, they mention that if you train the Reward Model for too long, the PPO agent eventually starts getting *worse* scores from actual humans, even though the Reward Model thinks it's doing great.
>
> **What is the name of this phenomenon?** (It's a specific 2-word term).
