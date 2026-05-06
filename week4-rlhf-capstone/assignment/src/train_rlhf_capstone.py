import argparse
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

# Note: The core PPO loops and architectures will be imported from Week 3
from ppo.ppo_trainer import PPOTrainer, TrainingConfig, get_hh_rewards


def load_target_and_reference_policies(actor_model_name: str) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Initialize the primary actor policy and freeze a copy as the reference policy.
    
    Args:
        actor_model_name: HuggingFace hub path or local directory to an SFT model.
        
    Returns:
        actor: The tunable policy $\pi_\theta$.
        ref: The frozen reference policy $\pi_{ref}$.
        tokenizer: Associated tokenizer.
    """
    # TODO: Implement parameter loading. Remember quantization strategies (e.g., 4-bit)
    # for large parameter spaces to avoid CUDA OOM.
    pass


def load_reward_model(reward_model_name: str) -> PreTrainedModel:
    r"""
    Initialize the preference scalar model (Critic $R_\phi$).
    
    Args:
        reward_model_name: HuggingFace hub path or local directory.
        
    Returns:
        critic: The frozen reward model.
    """
    # TODO: Load the scalar outcome model from Week 2.
    pass


def build_dataloader(dataset_name: str, batch_size: int, tokenizer: PreTrainedTokenizer) -> DataLoader:
    """
    Construct the data loader for sampling open-ended prompts.
    
    Args:
        dataset_name: Path to the prompt dataset.
        batch_size: Batch dimension for generation.
        tokenizer: Tokenizer to format sequences.
        
    Returns:
        DataLoader containing formatted prompt tensors.
    """
    # TODO: Construct prompt distribution.
    pass


config = TrainingConfig(
    model_name="Qwen/Qwen3-0.6B",
    reward_model_name="week2-reward-models/assignment/reward_model_checkpoint",
    dataset_name="Anthropic/hh-rlhf",
    prompt_token_len=64,
)


def execute_rlhf_loop(config: TrainingConfig) -> None:
    """
    The core PPO optimization loop.
    """
    trainer = PPOTrainer(config)
    print("Initializing RLHF Pipeline...")
    print("Generating responses...")
    generated_ids, gen_padding_masks, gen_output_masks, gen_texts_list = trainer.generate_responses()
    print("Rewarding responses...")
    rewards = get_hh_rewards(gen_texts_list, trainer.reward_model, trainer.reward_tokenizer, config.reward_batch_size)    # (num_prompts,)

    for i in range(len(gen_texts_list)):
        print(f"Response {i+1}: {gen_texts_list[i]}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reward: {rewards[i]} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("-"*100)



def main():
    # parser = argparse.ArgumentParser(description="Week 4 Capstone: Classic RLHF Pipeline")
    # parser.add_argument("--actor_model", type=str, required=True, help="Path to SFT model")
    # parser.add_argument("--reward_model", type=str, required=True, help="Path to Preference model")
    # parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf", help="Prompt dataset")
    # parser.add_argument("--beta", type=float, default=0.05, help="KL Divergence penalty coefficient")
    # parser.add_argument("--batch_size", type=int, default=8, help="Rollout batch size")
    # parser.add_argument("--epochs", type=int, default=1, help="Optimization epochs")
    # parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    
    # args = parser.parse_args()
    
    # print(rf"Initializing RLHF Pipeline with KL coefficient $\beta$ = {args.beta}...")

    config = TrainingConfig(
        model_name="Qwen/Qwen3-0.6B",
        reward_model_name="week2-reward-models/assignment/reward_model_checkpoint",
        # dataset
        dataset_name="Anthropic/hh-rlhf",
        prompt_token_len=32,
        num_prompts=16,             # rollout dataset size
        gen_batch_size=4,
        reward_batch_size=4,
        max_new_tokens=128,
    )
    
    execute_rlhf_loop(config)


if __name__ == "__main__":
    main()
