import pandas as pd
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


def load_imdb_dataset(split: str = "train") -> pd.DataFrame:
    """
    Loads the stanfordnlp/imdb dataset and returns it as a DataFrame.

    Args:
        split: Dataset split to load - "train" or "test".

    Returns:
        DataFrame with columns ["text", "label"] where label 0=neg, 1=pos.
    """
    dataset = load_dataset("stanfordnlp/imdb", split=split)
    return dataset.to_pandas()


def sample_prompt_batch(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    prompt_token_len: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples a batch of prompts from the dataset.
    Each prompt is truncated to the first `prompt_token_len` tokens,
    forming the initial context for the PPO generation rollout.

    Args:
        df: DataFrame returned by load_imdb_dataset().
        tokenizer: The tokenizer for the actor model.
        batch_size: Number of prompts to sample.
        prompt_token_len: Number of leading tokens to retain as the prompt.

    Returns:
        Tuple of (input_ids, attention_mask), each of shape (batch_size, prompt_token_len).
    """
    sample = df["text"].sample(n=batch_size).tolist()

    # Tokenize without padding first to extract the leading prompt tokens
    encoded = tokenizer(
        sample,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_token_len,
    )
    # Truncate to exactly prompt_token_len tokens
    input_ids = encoded["input_ids"][:, :prompt_token_len]
    attention_mask = encoded["attention_mask"][:, :prompt_token_len]

    return input_ids, attention_mask


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token

    df = load_imdb_dataset(split="train")
    print(f"Dataset size: {len(df)}")
    print(df.head(2))

    input_ids, attention_mask = sample_prompt_batch(df, tokenizer, batch_size=4)
    print(f"\nPrompt input_ids shape: {input_ids.shape}")   # (4, 8)
    print(f"Attention mask shape:   {attention_mask.shape}")

    # Decode to verify the prompts look like natural truncated text
    for i, ids in enumerate(input_ids):
        print(f"  Prompt {i}: {tokenizer.decode(ids)!r}")
