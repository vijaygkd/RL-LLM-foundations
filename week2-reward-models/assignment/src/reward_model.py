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
DATASET_NAME = "Anthropic/hh-rlhf"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
# TODO - update before full run
EPOCHS = 1
BATCH_SIZE = 4


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
    print("Training model...")
    print("Model name: ", MODEL_NAME)
    print("Dataset name: ", DATASET_NAME)
    print("Batch size: ", BATCH_SIZE)
    print("Epochs: ", EPOCHS)
    print("Device: ", DEVICE)

    model, tokenizer = init_model(MODEL_NAME)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    dataset = RLHFDataset(DATASET_NAME, "train", tokenizer)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Length of dataset: ", len(dataset))
    print("Length of dataloader: ", len(dataloader))

    # train loop
    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(dataloader):
            # forward pass
            winner_rewards, loser_rewards = forward_pass(model, batch)  # (B)
            # loss fn - Bradley-Terry (binary cross-entropy / NLL)
            loss = -torch.nn.functional.logsigmoid(winner_rewards - loser_rewards).mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # print loss every 10 batches
            if i % 10 == 0:
                print(f"Batch {i} loss: ", loss.item())
                break   # TODO - remove this break

        # run evaluation
        evaluate_model(model, tokenizer, no_of_batch=1)
    
    print("Training complete.")
    # full evaluation on test set
    print("-" * 30)
    print("Full evaluation on test set")
    evaluate_model(model, tokenizer, no_of_batch=10)
    print("Done.")

        
def forward_pass(model, batch):
    """
    Performs a forward pass through the model and returns the reward scores.
    """
    winner_batch, loser_batch = batch
    winner_output = model(
        input_ids=winner_batch[0].to(DEVICE), attention_mask=winner_batch[1].to(DEVICE)
    )   # (B, S, 1)
    loser_output = model(
        input_ids=loser_batch[0].to(DEVICE), attention_mask=loser_batch[1].to(DEVICE)
    )   # (B, S, 1)
    """
    # NOTE - Use output of last token as predicted sequence "reward" from RM.
    # The transformer outputs one scaler value for each input token position.
    # We use the last token's scaler output as the RM reward output
    # as the last position "sees" the entire text. 
    """
    # finding the last non-pad token. Sequence is right-padded.
    last_winner_token = (winner_batch[1].sum(dim=-1) - 1).unsqueeze(-1).unsqueeze(-1).to(DEVICE) # (B,1,1)
    last_loser_token = (loser_batch[1].sum(dim=-1) - 1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    # logits (B,S,1)
    # Gather scalar values from the exact last sequence index and squeeze to 1D array
    winner_rewards = winner_output.logits.gather(dim=1, index=last_winner_token).squeeze()    # (B)
    loser_rewards = loser_output.logits.gather(dim=1, index=last_loser_token).squeeze()       # (B)
    return winner_rewards, loser_rewards


def evaluate_model(model, tokenizer, no_of_batch=0):
    """
    Evaluate the Reward Model on test split of dataset
    """
    model.eval()
    dataset = RLHFDataset(DATASET_NAME, "test", tokenizer)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    accuracy = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader): 
            if no_of_batch > 0 and i >= no_of_batch:
                break
            winner_rewards, loser_rewards = forward_pass(model, batch)
            winner_rewards = winner_rewards.float().cpu().numpy()
            loser_rewards = loser_rewards.float().cpu().numpy()
            # NOTE - accuracy calculation as per Bradley-Terry preference objective
            accuracy_batch = (winner_rewards > loser_rewards).astype(int)
            accuracy.append(accuracy_batch)

    accuracy = np.concatenate(accuracy) 
    print("Accuracy: ", np.mean(accuracy))


if __name__ == "__main__":
    #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" # "Qwen/Qwen2-0.5B-Instruct" "google/gemma-3-1b-it"
    # model, tokenizer = init_model(MODEL_NAME)
    # evaluate_model(model, tokenizer)
    train_model()