import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class PromptsDataset(Dataset):
    """
    Wraps any HuggingFace text dataset, exposing each entry's raw text.
    Tokenization is deferred to the collate_fn so that dynamic padding
    is applied cleanly at the batch level.
    """

    def __init__(self, dataset_name: str, split: str = "train", text_column: str = "chosen"):
        dataset = load_dataset(dataset_name, split=split)
        self.texts = dataset[text_column]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        text = self.texts[idx]
        prompt_marker = "\n\nAssistant:"
        if prompt_marker in text:
            # Extract only the very first turn and drop the rest of the conversation
            prompt_parts = text.split(prompt_marker)
            return prompt_parts[0] + prompt_marker
        return text


class PromptCollator:
    """
    Collate function that tokenizes a batch of raw text strings and
    truncates each to `prompt_token_len` leading tokens.
    Passed directly to DataLoader as `collate_fn`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_token_len: int = 64):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.prompt_token_len = prompt_token_len

    def __call__(self, batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.prompt_token_len,
        )
        return encoded["input_ids"], encoded["attention_mask"]


def build_prompt_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "stanfordnlp/imdb",
    split: str = "train",
    text_column: str = "text",
    batch_size: int = 16,
    prompt_token_len: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience factory that wires together the Dataset, Collator, and DataLoader.

    Args:
        tokenizer: Actor model tokenizer. Must have pad_token set.
        dataset_name: Any HuggingFace dataset identifier with a text column.
        split: "train" or "test".
        batch_size: Number of prompts per batch.
        prompt_token_len: Number of leading tokens retained as the generation prompt.
        shuffle: Whether to shuffle the dataset each epoch.
        num_workers: Subprocess workers for data loading.

    Returns:
        A DataLoader yielding (input_ids, attention_mask) tuples of shape
        (batch_size, prompt_token_len).
    """
    dataset = PromptsDataset(dataset_name=dataset_name, split=split, text_column=text_column)
    collator = PromptCollator(tokenizer=tokenizer, prompt_token_len=prompt_token_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token

    loader = build_prompt_dataloader(tokenizer, dataset_name="stanfordnlp/imdb", batch_size=4, prompt_token_len=8)

    input_ids, attention_mask = next(iter(loader))
    print(f"input_ids shape:      {input_ids.shape}")       # (4, 8)
    print(f"attention_mask shape: {attention_mask.shape}")

    for i, ids in enumerate(input_ids):
        print(f"  Prompt {i}: {tokenizer.decode(ids)!r}")
