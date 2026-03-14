# Week 2 Practicum: Training the Reward Model

## Objective
* Train a Reward Model (RM) as a "Judge".
* Replace CartPole's hardcoded rewards with a parameterized model mimicking human preference.

## Mathematical Objective
* Implement the Bradley-Terry preference model.
* Assign a higher scalar reward to the chosen ($y_w$) vs. rejected ($y_l$) response for a given prompt ($x$).
* Minimize the ranking loss:
$$\mathcal{L}_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l))$$

## Assignment Steps

### Step 1: Dataset Preparation
* Load the `Anthropic/hh-rlhf` dataset.
* Parse raw dialogue strings into a shared prompt ($x$), a chosen continuation ($y_w$), and a rejected continuation ($y_l$).

### Step 2: Model Architecture
* Use a small transformer base (e.g., `DistilBERT` or `TinyLlama`).
* Modify the architecture to output a single scalar value.
* Project the final hidden state of the concatenated $(x, y)$ sequence into a 1D reward score $r(x, y)$.

### Step 3: Training Loop
* Implement the custom Bradley-Terry loss function.
* Forward pass: compute rewards for both chosen and rejected sequences independently.
* Backpropagate the log-sigmoid of their difference.

### Step 4: Evaluation
* Evaluate the model's ability to predict human preference on the test split.
* A prediction is correct if $r(x, y_w) > r(x, y_l)$.

## Success Criteria
* Dataset is successfully processed into chosen/rejected pairs.
* Custom loss strictly follows the Bradley-Terry formulation.
* The trained model achieves **>65% accuracy** on the test set.
