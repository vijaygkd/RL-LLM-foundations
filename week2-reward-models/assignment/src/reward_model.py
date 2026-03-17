"""
Train a reward model using the Bradley-Terry model
on a dataset of (promt + completion) pairs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RLHFDataset, create_dataloader

from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def init_model(model_name: str):
    """
    Load Base / SFT checkpoint model and 
    replace last unembedding layer with single linear output layer.
    """
    print("Loading model: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.lm_head = nn.Linear(model.lm_head.in_features, 1)
    print(model)
    return model.to(DEVICE, dtype=torch.bfloat16), tokenizer


def train_model():
    """
    Train the model
    """
    pass


def evaluate_model(model, tokenizer):
    """
    Evaluate the Reward Model on test split of dataset
    """
    dataset = RLHFDataset("Anthropic/hh-rlhf", "test", tokenizer)
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=False)

    print("Length of dataset: ", len(dataset))
    print("Length of dataloader: ", len(dataloader))

    accuracy = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            winner_batch, loser_batch = batch
            winner_output = model(
                input_ids=winner_batch[0].to(DEVICE), attention_mask=winner_batch[1].to(DEVICE)
            )
            loser_output = model(
                input_ids=loser_batch[0].to(DEVICE), attention_mask=loser_batch[1].to(DEVICE)
            )
            """
            # NOTE - Use output of last token as predicted "reward" from RM.
            The transformer outputs one scaler value for each input token position.
            We use the last token's scaler output as the RM reward output
            as it "sees" the entire text. 
            """
            # finding the last non-pad token. Sequence is right-padded.
            last_winner_token = (winner_batch[1].sum(dim=-1) - 1).unsqueeze(-1).unsqueeze(-1).to(DEVICE) # (B,1,1)
            last_loser_token = (loser_batch[1].sum(dim=-1) - 1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
            # logits (B,S,1)
            # Gather scalar values from the exact last sequence index and squeeze to 1D array
            winner_rewards = winner_output.logits.gather(dim=1, index=last_winner_token).squeeze().float().cpu().numpy()    # (B)
            loser_rewards = loser_output.logits.gather(dim=1, index=last_loser_token).squeeze().float().cpu().numpy()       # (B)
            # NOTE - accuracy calculation as per Bradley-Terry preference objective
            accuracy_batch = (winner_rewards > loser_rewards).astype(int)
            accuracy.append(accuracy_batch)
            print(f"Batch {i} accuracy: ", np.mean(accuracy_batch))

    accuracy = np.concatenate(accuracy) 
    print("Accuracy: ", np.mean(accuracy))
 


if __name__ == "__main__":
    #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" # "Qwen/Qwen2-0.5B-Instruct" "google/gemma-3-1b-it"
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model, tokenizer = init_model(model_name)
    evaluate_model(model, tokenizer)