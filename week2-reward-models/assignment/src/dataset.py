from typing import Dict, Any, List, Tuple
import math
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class RLHFDataset(Dataset):
    """
    Dataset for training a Reward Model using the Anthropic/hh-rlhf dataset.
    """
    def __init__(self, data_path: str, split: str, tokenizer: Any, max_length: int = 768):
        """
        Initializes the dataset.

        Args:
            data_path: Path to the dataset (or HuggingFace dataset name).
            split: The split to load.
            tokenizer: The tokenizer to use for encoding text.
            max_length: Maximum sequence length for the transformer.
        """
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()

    def _load_data(self) -> Any:
        
        # Load the dataset using the HuggingFace datasets library
        dataset = load_dataset(self.data_path)
        return dataset[self.split]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single sample from the dataset.
        
        A sample should contain the tokenized prompt + chosen response, 
        and the tokenized prompt + rejected response.
        """
        item = self.data[idx]
        chosen_text = item['chosen']
        rejected_text = item['rejected']

        if self.tokenizer is None:
            return {'chosen_text': chosen_text, 'rejected_text': rejected_text}

        # Tokenize chosen and rejected sequences. 
        # We don't pad here; we will pad dynamically in the collate function.
        chosen_enc = self.tokenizer(
            chosen_text, 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        rejected_enc = self.tokenizer(
            rejected_text, 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True
        )

        return {
            'chosen_input_ids': chosen_enc['input_ids'],
            'chosen_attention_mask': chosen_enc['attention_mask'],
            'rejected_input_ids': rejected_enc['input_ids'],
            'rejected_attention_mask': rejected_enc['attention_mask']
        }


class RLHFCollateFn:
    def __init__(self, pad_token_id: int, pad_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_multiple_of = pad_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Collates a batch of tokenized sequences into padded tensors.
        
        Returns two pairs of tensors:
        winner_batch: (chosen_input_ids, chosen_attention_mask)
        loser_batch: (rejected_input_ids, rejected_attention_mask)
        """
        # Find the maximum sequence length in the current batch
        max_len = 0
        for item in batch:
            max_len = max(max_len, len(item['chosen_input_ids']), len(item['rejected_input_ids']))
        
        # Round up max_len to the nearest multiple for GPU efficiency
        max_len = math.ceil(max_len / self.pad_multiple_of) * self.pad_multiple_of

        chosen_ids_batch = []
        chosen_mask_batch = []
        rejected_ids_batch = []
        rejected_mask_batch = []

        for item in batch:
            c_ids = item['chosen_input_ids']
            c_mask = item['chosen_attention_mask']
            r_ids = item['rejected_input_ids']
            r_mask = item['rejected_attention_mask']
            
            # Pad chosen
            c_pad_len = max_len - len(c_ids)
            chosen_ids_batch.append(c_ids + [self.pad_token_id] * c_pad_len)
            chosen_mask_batch.append(c_mask + [0] * c_pad_len)
            
            # Pad rejected
            r_pad_len = max_len - len(r_ids)
            rejected_ids_batch.append(r_ids + [self.pad_token_id] * r_pad_len)
            rejected_mask_batch.append(r_mask + [0] * r_pad_len)

        # Return winner and loser batches. You may need attention masks too for the transformer.
        # Format: winner_batch=(ids, mask), loser_batch=(ids, mask)
        winner_batch = (
            torch.tensor(chosen_ids_batch, dtype=torch.long),
            torch.tensor(chosen_mask_batch, dtype=torch.long)
        )
        loser_batch = (
            torch.tensor(rejected_ids_batch, dtype=torch.long),
            torch.tensor(rejected_mask_batch, dtype=torch.long)
        )
        
        return winner_batch, loser_batch


def create_dataloader(dataset: RLHFDataset, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the given dataset.
    """
    # Fallback to 0 if tokenizer is unavailable
    if dataset.tokenizer:
        pad_token_id = dataset.tokenizer.pad_token_id
    else:
        pad_token_id = 0
    
    collate_fn = RLHFCollateFn(pad_token_id=pad_token_id, pad_multiple_of=8)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )

def debug_dataset():
    import random
    # load the dataset train split
    dataset = RLHFDataset("Anthropic/hh-rlhf", "train", None)
    
    print("Length of dataset: ", len(dataset))
    random_index = random.randint(0, len(dataset)-1)
    
    sample = dataset[random_index]
    
    print("-" * 30)
    print("Chosen response:\n", sample['chosen_text'])
    print("-" * 30)
    print("Rejected response:\n", sample['rejected_text'])


# ============================================================================
# UTILITIES - Dataset analysis helpers (not part of core training pipeline)
# ============================================================================

def analyze_sequence_lengths(
    data_path: str = "Anthropic/hh-rlhf",
    tokenizer_name: str = "Qwen/Qwen2-0.5B-Instruct",
    sample_size: int = 10000,
):
    """
    Analyzes the token sequence length distribution of the dataset.
    Use this to determine an appropriate max_length for truncation.

    Results (Qwen2-0.5B-Instruct tokenizer, 10k sample):
        p50=165, p90=423, p95=530, p99=835, max=1964
        <= 512: 94.3%  |  <= 768: 98.6%  |  <= 1024: 99.5%
    """
    import random
    import numpy as np
    from transformers import AutoTokenizer
    from tqdm import tqdm

    print("Loading dataset and tokenizer...")
    dataset = load_dataset(data_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    subset = dataset.select(indices)

    lengths = []
    for item in tqdm(subset, desc="Tokenizing"):
        c_len = len(tokenizer.encode(item["chosen"], add_special_tokens=True))
        r_len = len(tokenizer.encode(item["rejected"], add_special_tokens=True))
        lengths.extend([c_len, r_len])

    lengths = np.array(lengths)
    print(f"\n--- Sequence Length Distribution ({len(lengths)} sequences) ---")
    print(f"  Min:  {np.min(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  p50:  {np.percentile(lengths, 50):.0f}")
    print(f"  p75:  {np.percentile(lengths, 75):.0f}")
    print(f"  p90:  {np.percentile(lengths, 90):.0f}")
    print(f"  p95:  {np.percentile(lengths, 95):.0f}")
    print(f"  p99:  {np.percentile(lengths, 99):.0f}")
    print(f"  Max:  {np.max(lengths)}")
    print(f"\n  % sequences <= 512:  {(lengths <= 512).mean()*100:.1f}%")
    print(f"  % sequences <= 768:  {(lengths <= 768).mean()*100:.1f}%")
    print(f"  % sequences <= 1024: {(lengths <= 1024).mean()*100:.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_sequence_lengths()
    else:
        debug_dataset()