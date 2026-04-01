from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from ppo_model import PPOModel
from dataset import build_prompt_dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class TrainingConfig:
    model_name: str
    reward_model_name: str
    # dataset
    dataset_name: str = "stanfordnlp/imdb"
    num_prompts: int = 8 #512             # TODO - ?? Might be better to generate large batch of rollouts in one go 
                                        # to save inference time and reduce GPU wall-time. 
                                        # Or should we do it in batches..?
    gen_batch_size: int = 4 # 32
    reward_batch_size: int = 4
    num_rollouts_per_prompt: int = 1    # TODO - Do we need multiple rollouts per prompt? 
    prompt_token_len: int = 8
    max_new_tokens: int = 24            # T_total = T_prompt + T_new = 8 + 24 = 32
    # training
    ppo_steps: int = 1000
    micro_steps: int = 4                # number of gradient updates per PPO step --> not needed maybe
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4    # effective batch size = micro_batch_size * gradient_accumulation_steps = 16
    # hyper-parameters
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.1
    # optimizers
    lr: float = 1e-6
    # logging
    log_freq: int = 10
    save_freq: int = 100


class PPOTrainer:
    """
    PPO Trainer class
    1. Init 4 models (actor, critic, ref, reward)
    2. Load dataset - stanfordnlp/imdb (first 8 tokens as prompts only)
    3. Run PPO training loop for N ppo_steps
    """
    def __init__(self, config: TrainingConfig):
        print("Using device: ", DEVICE)
        self.config = config
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print("Tokenizer pad token id: ", self.text_tokenizer.pad_token_id)
        print("Tokenizer eos token id: ", self.text_tokenizer.eos_token_id)
        # PPO models setup
        # -------------------------------------
        print("Loading PPO model... ", config.model_name)
        self.ppo_model = PPOModel(config.model_name, add_value_head=True).to(DEVICE)                                # actor + critic    
        print("Loading reference model... ", config.model_name)
        self.ref_model = PPOModel(config.model_name, add_value_head=False).to(DEVICE).requires_grad_(False)         # reference model (frozen)
        # reward model setup
        # -------------------------------------
        print("Loading reward model... ", config.reward_model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
        # TODO - can this load the reward model trained in week 2?
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_name).to(DEVICE).requires_grad_(False)
        # dataset setup
        # -------------------------------------
        print("Loading dataset... ", config.dataset_name)
        self.prompt_dataloader = build_prompt_dataloader(
            tokenizer=self.text_tokenizer,
            dataset_name=config.dataset_name,
            batch_size=config.gen_batch_size,
            prompt_token_len=config.prompt_token_len,
            shuffle=True,
            num_workers=4
        )

    def generate_responses(self):
        """
        Generates rollouts based on prompts using current policy
        Returns: (generated_ids, generated_attn_masks, gen_texts_list)
        generated_ids: Generated token ids (num_prompts, T_total)
        generated_attn_masks: Generated attention masks, (num_prompts, T_total) 
        gen_texts_list: list of generated texts (num_prompts,)
        """
        num_generation_batches = self.config.num_prompts // self.config.gen_batch_size
        assert num_generation_batches * self.config.gen_batch_size == self.config.num_prompts, "num_prompts must be divisible by gen_batch_size"
        
        gen_ids_list, gen_attn_masks_list, gen_texts_list = [], [], []
        for i in range(num_generation_batches):
            input_ids, input_attn_mask = next(iter(self.prompt_dataloader))
            # Generate rollouts with "current" policy
            with torch.no_grad():
                # autoregressive generation
                generation_ids = self.ppo_model.actor.generate(             # (B, T) = (B, T_prompt + T_new)
                    input_ids.to(DEVICE),                                   # (B, T_prompt)
                    attention_mask=input_attn_mask.to(DEVICE),
                    do_sample=True, 
                    max_new_tokens=self.config.max_new_tokens)
                generated_texts = self.text_tokenizer.batch_decode(
                    generation_ids, skip_special_tokens=True
                )
                # mask padding tokens - Left and Right padding
                gen_attn_mask = generation_ids != self.text_tokenizer.pad_token_id      # (B, T)

                gen_ids_list.append(generation_ids)
                gen_attn_masks_list.append(gen_attn_mask)
                gen_texts_list.extend(generated_texts)

        generated_ids = torch.cat(gen_ids_list, dim=0)                      # (num_prompts, T)
        generated_attn_masks = torch.cat(gen_attn_masks_list, dim=0)        # (num_prompts, T)
        
        assert generated_ids.shape == (self.config.num_prompts, self.config.prompt_token_len + self.config.max_new_tokens)
        assert generated_attn_masks.shape == generated_ids.shape
        assert len(gen_texts_list) == self.config.num_prompts

        return generated_ids, generated_attn_masks, gen_texts_list


    def get_log_probs_and_values(self, model: PPOModel, generation_ids: torch.Tensor, generation_attn_masks: torch.Tensor):
        """
        Compute log probs and values for generated tokens
        generation_ids: (B, T) T = T_prompt + T_new
        generation_attn_masks: (B, T)   padding mask
        return 
            log_probs:      (B, T-1) left-shifted by 1
            critic_output:  (B, T)   values for each token in the sequence
        """
        lm_output, critic_output = model(generation_ids, generation_attn_masks)             # (B, T, V), (B, T, 1) or None
        # left shift by 1 for autoregressive generation
        logits = lm_output.logits[:, :-1, :]                                               # (B, T-1, V) 
        labels = generation_ids[:, 1:].unsqueeze(-1)                                       # (B, T-1, 1) 
        # calculate log-prob
        log_probs = torch.log_softmax(logits, dim=-1)                                      # (B, T-1, V) 
        log_probs = torch.gather(log_probs, dim=-1, index=labels)                          # (B, T-1, 1)
        log_probs = log_probs.squeeze(-1)                                                  # (B, T-1)
        # critic values must align with generated tokens / actions (T-1 tokens)
        # a.) the log_probs are T-1 because of left shifting - last position output is ignored because no label. 
        # b.) we keep T-1 (aligned with actions / token log_probs) and 1 last extra for bootstrapping GAE in case it's not EOS (terminal state)
        if critic_output is not None:
            critic_output = critic_output.squeeze(-1)                                      # (B, T)

        assert log_probs.shape == (generation_ids.shape[0], generation_ids.shape[1] - 1)
        if critic_output is not None:
            assert critic_output.shape == (generation_ids.shape[0], generation_ids.shape[1])

        return log_probs, critic_output


    def compute_advantages(self):
        # TODO- ignore prompt tokens in GAE and loss calculation
        # TODO - Ignore padding tokens in GAE calculation
        # TODO - Ignore values for input tokens in GAE calculation
        # gen_attn_mask[:, :self.config.prompt_token_len] = 0
        pass

    def train(self):
        """
        3. Run PPO training loop for N ppo_steps
            a. Generate responses from actor from dataset prompts           --> TODO: how many rollouts responses per prompt?
            b. Compute sequence-level rewards using static reward model
            for batch_size in generate_responses():
                for micro_step in range(num_micro_steps):                   --> TODO: should we run multiple steps over same mini-batch? or just 1?
                    c. Calculate timestep (token-level) KL divergence between actor and ref
                    d. Compute log-prob from PPO LM model (actor, critic, ref_model)
                    e. Compute advantage estimates (GAE) at token-level using reward, KL, value
                    f. Compute PPO loss = PPO clipped loss + beta * KL divergence penalty 
                    g. Update actor and critic: Loss = PPO loss + value loss (MSE)
        """
        pass

########################################
# Reward functions
# TODO - move this to a separate file because we might want to use different reward functions.
# PPOTrainer should be reward-agnostic
########################################
def get_sentiment_rewards(generated_texts: list[str], reward_model, reward_tokenizer, reward_batch_size=4):
    """
    Return scalar sentiment reward for each generated text
    Reward = Degree of positivity
    """
    assert len(generated_texts) % reward_batch_size == 0, "Number of generated texts must be divisible by reward_batch_size"
    rewards = []
    with torch.no_grad():
        for i in range(0, len(generated_texts), reward_batch_size):
            encoded = reward_tokenizer(
                generated_texts[i:i+reward_batch_size], 
                return_tensors="pt", 
                padding=True, 
                truncation=True).to(DEVICE)
            outputs = reward_model(**encoded).logits            # (B, 3)    Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
            positive_score = outputs[:, 2]                      # (B,)      # TODO - should we return prob or logits?
            rewards.extend(positive_score.cpu().tolist())
    return rewards


if __name__ == "__main__":
    config = TrainingConfig(
        model_name="Qwen/Qwen3-0.6B",
        reward_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    trainer = PPOTrainer(config)
    generated_ids, generated_attn_masks, gen_texts_list = trainer.generate_responses()
    log_prbs, critic_output = trainer.get_log_probs_and_values(trainer.ppo_model, generated_ids, generated_attn_masks)
    ref_log_prbs, _ = trainer.get_log_probs_and_values(trainer.ref_model, generated_ids, generated_attn_masks)
    rewards = get_sentiment_rewards(gen_texts_list, trainer.reward_model, trainer.reward_tokenizer, config.reward_batch_size)
    for i in range(len(gen_texts_list)):
        print("Score: ", rewards[i], "Text: ", gen_texts_list[i])
