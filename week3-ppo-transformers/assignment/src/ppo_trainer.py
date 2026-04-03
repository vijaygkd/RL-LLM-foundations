from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from ppo_model import PPOModel
from dataset import build_prompt_dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class TrainingConfig:
    model_name: str
    reward_model_name: str
    # dataset
    dataset_name: str               = "stanfordnlp/imdb"
    num_prompts: int                = 512             # rollout dataset size
    gen_batch_size: int             = 64               
    reward_batch_size: int          = 64
    prompt_token_len: int           = 8
    max_new_tokens: int             = 24            # T_total = T_prompt + T_new = 8 + 24 = 32
    # training
    ppo_epochs: int                 = 100
    learning_epochs: int            = 4             # number of learning epoch per set of rollouts --> InstructGPT paper uses 4
    batch_size: int                 = 16             # learning loop batch size
    grad_accumulation_steps: int    = 4             # effective batch size = batch_size * gradient_accumulation_steps = 64
    # hyper-parameters
    clip_epsilon: float             = 0.2
    kl_beta: float                  = 0.01
    gae_gamma: float                = 0.99
    gae_lambda: float               = 0.95
    value_loss_coef: float          = 1
    # optimizers
    lr: float                       = 1e-6
    max_grad_norm: float            = 1.0
    # logging
    log_freq: int                   = 10
    save_freq: int                  = 100


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
        # -------------------------------------
        # PPO models setup
        print("Loading PPO model... ", config.model_name)
        self.ppo_model = PPOModel(config.model_name, add_value_head=True).to(DEVICE)                                # actor + critic    
        print("Loading reference model... ", config.model_name)
        self.ref_model = PPOModel(config.model_name, add_value_head=False).to(DEVICE).requires_grad_(False)         # reference model (frozen)
        # -------------------------------------
        # reward model setup
        print("Loading reward model... ", config.reward_model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
        # TODO - can this load the reward model trained in week 2?
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_name).to(DEVICE).requires_grad_(False)
        # -------------------------------------
        # Dataset
        print("Loading dataset... ", config.dataset_name)
        self.prompt_dataloader = build_prompt_dataloader(
            tokenizer=self.text_tokenizer,
            dataset_name=config.dataset_name,
            batch_size=config.gen_batch_size,
            prompt_token_len=config.prompt_token_len,
            shuffle=True,
            num_workers=4,
            split="train[:32]"
        )

    def generate_responses(self):
        """
        Generates rollouts based on prompts using current policy
        Returns: (generated_ids, generated_attn_masks, gen_texts_list)
            generated_ids: Generated token ids (num_prompts, T_total)
            generated_padding_masks: Generated padding masks, (num_prompts, T_total) 
            generated_output_mask: Generated output mask, (num_prompts, T_total)  1: generated token, 0: prompt or padding token
            gen_texts_list: list of generated texts (num_prompts,)
        """
        target_len = self.config.prompt_token_len + self.config.max_new_tokens
        num_generation_batches = self.config.num_prompts // self.config.gen_batch_size
        assert num_generation_batches * self.config.gen_batch_size == self.config.num_prompts, "num_prompts must be divisible by gen_batch_size"
        
        gen_ids_list, gen_padding_masks_list, gen_output_mask_list, gen_texts_list = [], [], [], []
        for i in range(num_generation_batches):
            input_ids, input_attn_mask = next(iter(self.prompt_dataloader))
            # Generate rollouts with "current" policy
            with torch.no_grad():
                # autoregressive generation
                generation_ids = self.ppo_model.actor.generate(             # (B, T) = (B, T_prompt + T_new)
                    input_ids.to(DEVICE),                                   # (B, T_prompt)
                    attention_mask=input_attn_mask.to(DEVICE),              # can be left padded during generation
                    do_sample=True, 
                    max_new_tokens=self.config.max_new_tokens)
                generated_texts = self.text_tokenizer.batch_decode(
                    generation_ids, skip_special_tokens=True
                )
                # mask padding tokens - Left and Right padding - 1: non-padding token, 0: padding token
                gen_padding_mask = (generation_ids != self.text_tokenizer.pad_token_id).long()        # (B, T)
                # Output tokens mask - 1: generated token, 0: prompt or padding token
                gen_output_mask = gen_padding_mask.clone()            # right padding
                gen_output_mask[:, :input_ids.shape[1]-1] = 0         # (left padding + prompt tokens) are masked except last token where first token is generated

                # right pad to target length
                pad_len = target_len - generation_ids.shape[1]
                if pad_len > 0:
                    # F.pad format for 1D/2D: (pad_left, pad_right) for the last dimension
                    generation_ids = F.pad(generation_ids, (0, pad_len), value=self.text_tokenizer.pad_token_id)
                    gen_padding_mask = F.pad(gen_padding_mask, (0, pad_len), value=0)
                    gen_output_mask = F.pad(gen_output_mask, (0, pad_len), value=0)

                gen_ids_list.append(generation_ids)
                gen_padding_masks_list.append(gen_padding_mask)
                gen_output_mask_list.append(gen_output_mask)
                gen_texts_list.extend(generated_texts)

        generated_ids = torch.cat(gen_ids_list, dim=0)                            # (num_prompts, T)
        generated_padding_masks = torch.cat(gen_padding_masks_list, dim=0)        # (num_prompts, T) 
        generated_output_mask = torch.cat(gen_output_mask_list, dim=0)            # (num_prompts, T) 
        
        assert generated_ids.shape == (self.config.num_prompts, self.config.prompt_token_len + self.config.max_new_tokens)
        assert generated_padding_masks.shape == generated_ids.shape
        assert generated_output_mask.shape == generated_ids.shape
        assert len(gen_texts_list) == self.config.num_prompts

        return generated_ids, generated_padding_masks, generated_output_mask, gen_texts_list


    def get_log_probs_and_values(self, model: PPOModel, generation_ids: torch.Tensor, generation_attn_masks: torch.Tensor):
        """
        Compute log probs and values for generated tokens
        generation_ids: (B, T) T = T_prompt + T_new
        generation_attn_masks: (B, T)   padding mask
        return 
            log_probs:      (B, T-1) left-shifted by 1
            critic_values:  (B, T)   values for each token in the sequence
        """
        lm_output, critic_values = model(generation_ids, generation_attn_masks)             # (B, T, V), (B, T, 1) or None
        
        # left shift by 1 for autoregressive generation
        logits = lm_output.logits[:, :-1, :]                                               # (B, T-1, V) 
        labels = generation_ids[:, 1:]                                                    # (B, T-1) 
        # calculate log-prob for labels
        # TIP: use F.cross_entropy with reduction="none" and multiply by -1 to get log_probs to save memory
        # see note softmax_optimization_note.md
        # F.cross_entropy expects classes/vocab to be in the second dimension: (B, V, T-1)
        logits_transposed = logits.transpose(1, 2)
        # Compute fused -log(P(target))
        log_probs = -F.cross_entropy(logits_transposed, labels, reduction="none") # -> Output: (B, T-1)
        
        # critic values must align with generated tokens / actions (T-1 tokens)
        # a.) the log_probs are T-1 because of left shifting - last position output is ignored because no label. 
        # b.) we keep T-1 (aligned with actions / token log_probs) and 1 last extra for bootstrapping GAE in case it's not EOS (terminal state)
        if critic_values is not None:
            critic_values = critic_values.squeeze(-1)                                      # (B, T)

        assert log_probs.shape == (generation_ids.shape[0], generation_ids.shape[1] - 1)
        if critic_values is not None:
            assert critic_values.shape == (generation_ids.shape[0], generation_ids.shape[1])

        return log_probs, critic_values

    
    def compute_kl_token_penalty(self, log_probs_actor, log_probs_ref):
        """
        log_probs_actor: (B, T-1)
        log_probs_ref: (B, T-1)
        Returns
            kl_token_penalty: (B, T-1)
            token-level KL divergence for each token in the sequence
            KL(P || Q) = log(P(x)/Q(x)) = log P(x) - log Q(x)
        """
        kl_token_penalty = log_probs_actor - log_probs_ref          # (B, T-1)
        return kl_token_penalty


    def compute_gae_advantages(self, sequence_rewards, kl_token_penalty, critic_values, gen_output_mask):
        """
        Calculate per-timestep (token) advantage using GAE
        Input:
            sequence_rewards: (B,)      reward per sequence
            kl_token_penalty: (B, T-1)  token level kl-penalty
            critic_values: (B, T)       per-token critical value +1 last token for bootstrapping
            gen_output_mask: (B, T)     1: generated token, 0: prompt or padding token
        Returns:
            advantages: (B, T-1)        advantages for each token in the sequence expect last token used for bootstrapping (not used in loss)
        """

        # psuedo code:
        # ------------------------------------------
        # adv = [0]s
        # a_next = 0 if T is terminal else V_T
        # t from last timestamp T-1 -> 0    
        #     R = seq_reward if t is terminal or t is last else 0
        #     r_t = R - beta * kl_penalty
        #     td_t = (r_t + gamma * V_t+1)) - V_t
        #     a_t = td_t + gamme * lambda * a_next
        #     a_next = a_t 
        # ------------------------------------------

        seq_len = gen_output_mask.shape[1]                                                 # T
        # Find position of last output token
        first_output_token_idx = gen_output_mask.flip(dims=[1]).argmax(dim=1)              # (B,)
        last_output_token_idx = (seq_len-1) - first_output_token_idx                       # (B,)
        last_output_token_idx = torch.min(
            last_output_token_idx, 
            torch.ones_like(last_output_token_idx) * (seq_len - 2)                         # we ignore last buffer token, last output token for GAE is T-2
        )                                                                                  # (B,)

        # Calculate Rewards = R_t - beta * KL_t
        rewards = torch.zeros_like(kl_token_penalty)                # (B, T-1)
        rewards.scatter_(
            dim=1,
            index=last_output_token_idx.unsqueeze(-1),
            src=sequence_rewards.unsqueeze(-1).to(rewards.dtype),
        )                                                           # (B, T-1)  set reward only at last output token position
        kl_penalty = kl_token_penalty * gen_output_mask[:, :-1]     # (B, T-1)  zero out kl penalty for prompt and padding tokens
        rewards -=  self.config.kl_beta * kl_penalty                # (B, T-1)  subtract KL penalty for each token
        rewards *= gen_output_mask[:, :-1]                          # (B, T-1)  zero out rewards for prompt and padding tokens

        # Calculate TD errors
        masked_values = critic_values * gen_output_mask                            # (B, T)    zero out critic values for terminal states
        td_errors = ((
            rewards                                                 # reward at t
            + self.config.gae_gamma * masked_values[:, 1:])         # future value (right shifted)
            - masked_values[:, :-1])                                # expected value
        td_errors *= gen_output_mask[:, :-1]                        # (B, T-1)  zero out TD errors for prompt and padding tokens

        # Calculate GAE advantages - recursive calculation at each t from T-1 to 0
        advantages = torch.zeros_like(td_errors)                                                # (B, T-1)
        # IMP - init a_next after last output token as 0
        a_next = torch.zeros(critic_values.shape[0], device=DEVICE)                             # (B, 1)    use last token value for bootstrapping if not terminal
        for t in range(seq_len - 2, -1, -1):
            # calculation vectorized over timestamp dimension across batches
            a_t = td_errors[:, t] + self.config.gae_gamma * self.config.gae_lambda * a_next     # (B, 1) at t
            a_t = a_t * gen_output_mask[:, t]                                                   # (B, 1)    zero out a_t for prompt and padding tokens
            a_next = a_t                                                                        # (B, 1)    update a_next for next iteration
            advantages[:, t] = a_t.squeeze(-1)                                                  # update (B, T-1)  at t

        assert advantages.shape == kl_token_penalty.shape                                       # (B, T-1)
        return advantages                                                                       # (B, T-1)
        
        

    def train(self):
        """
        PPO training loop -- generate rollouts, and update Actor and Critic models using PPO loss
        """

        # Psuedo code:
        # Run PPO training loop for N ppo_steps
        #     (step 1- 4: No Grad)
        #     1. Generate responses from actor from dataset prompts
        #     2. Compute sequence-level rewards using static reward model
        #     3. Old policy log-prob and values from PPO model (actor, critic)                                
        #     4. Compute Advantages using GAE (using rewards, KL, old values)
        #     (step 5: w/ Grad)
        #     5. for num of learning epochs:
        #         for batch in generated data:
        #             a. Compute current policy log-prob and values from PPO model (actor, critic)
        #             b. Compute PPO loss = PPO clipped loss + Value Loss (MSE) 
        #             c. Backprop
        print("Starting PPO training...")
        num_update_steps = self.config.ppo_epochs * self.config.learning_epochs * self.config.num_prompts / self.config.batch_size  // (self.config.batch_size * self.config.grad_accumulation_steps)
        print(f"Number of learning steps: {num_update_steps} = Epochs {self.config.ppo_epochs} * Learning Epochs {self.config.learning_epochs} * Prompts {self.config.num_prompts} / (Batch Size {self.config.batch_size} * Gradient Accumulation Steps {self.config.grad_accumulation_steps})")

        # Initialize optimizer - actor and critic parameters
        optimizer = torch.optim.Adam(self.ppo_model.parameters(), lr=self.config.lr)
        optimizer.zero_grad()
        
        # Rollout generation loop
        for ppo_epoch in range(self.config.ppo_epochs):
            # 1. Generate responses using Old policy from dataset prompts
            print(f"Generating fresh rollouts [{self.config.num_prompts}] ...")
            generated_ids, gen_padding_masks, gen_output_masks, gen_texts_list = self.generate_responses()    # (num_prompts, T), (num_prompts, T), (num_prompts, T), (num_prompts,)
            # 2. Get rewards using static reward model for generated sequences
            print("Getting rewards ...")
            rewards = get_sentiment_rewards(gen_texts_list, self.reward_model, self.reward_tokenizer, self.config.reward_batch_size)    # (num_prompts,)

            num_steps = len(generated_ids) // self.config.batch_size
            # 3. Calculate Old and Ref policy log-prob and values for entire rollouts
            print("Calculating Old and Ref policy log-prob and values ...")
            old_log_probs_list = []
            old_critic_values_list = []
            ref_log_probs_list = []
            with torch.no_grad():                   # Imp - Gradients only flow through on-policy model (ppo_model)
                for i in range(num_steps):
                    batch_idx = i * self.config.batch_size
                    batch_generated_ids = generated_ids[batch_idx:batch_idx+self.config.batch_size]                 # (B, T)
                    batch_gen_padding_masks = gen_padding_masks[batch_idx:batch_idx+self.config.batch_size]         # (B, T)
                    batch_gen_output_masks = gen_output_masks[batch_idx:batch_idx+self.config.batch_size]           # (B, T)
                    batch_rewards = rewards[batch_idx:batch_idx+self.config.batch_size]                             # (B,)
                    # forward pass - no grad!
                    old_log_probs, old_critic_values = self.get_log_probs_and_values(                               # (B, T-1), (B, T)
                        self.ppo_model,
                        batch_generated_ids,
                        batch_gen_padding_masks
                    )
                    ref_log_probs, _ = self.get_log_probs_and_values(                                               # (B, T-1)
                        self.ref_model,
                        batch_generated_ids,
                        batch_gen_padding_masks
                    )
                    old_log_probs_list.append(old_log_probs)
                    old_critic_values_list.append(old_critic_values)
                    ref_log_probs_list.append(ref_log_probs)
            # concatenate all batches
            old_log_probs = torch.cat(old_log_probs_list, dim=0)                                                    # (num_prompts, T-1)
            old_critic_values = torch.cat(old_critic_values_list, dim=0)                                            # (num_prompts, T)
            ref_log_probs = torch.cat(ref_log_probs_list, dim=0)                                                    # (num_prompts, T-1)

            # 4. Calculate Advantages using GAE 
            # PRO-TIP: In RL, rewards, KL-penalty and Advantages are static observations of the environment. (No gradient should flow through environment)
            # These are computed once per rollout using the Old policy and are not updated during policy learning. (Check CHRONICLES.md)
            print("Calculating Advantages ...")
            with torch.no_grad():
                kl_penalty = self.compute_kl_token_penalty(                                                         # (num_prompts, T-1)
                    old_log_probs, ref_log_probs,
                )
                advantages = self.compute_gae_advantages(                                                               # (num_prompts, T-1)
                    rewards,                                                                                        # (num_prompts,)
                    kl_penalty,                                                                                     # (num_prompts, T-1)
                    old_critic_values,                                                                              # (num_prompts, T)
                    gen_output_masks                                                                                # (num_prompts, T)
                )
                # target for value loss
                returns = advantages + old_critic_values[:, :-1]                                                    # (num_prompts, T-1)
                # IMPORTANT: normalize advantages - used for PPO clipped loss, not for value loss
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # TODO - log time spent during rollout generation vs policy training. Might reveal bottleneck in PPO/RLHF training.
            
            # 5. Policy Learning loop
            for learning_epoch in range(self.config.learning_epochs):                   # epochs over same batch of rollouts
                print(f"Policy Learning Epoch {learning_epoch+1}/{self.config.learning_epochs}...")
                
                for i in range(num_steps):    
                    batch_idx = i * self.config.batch_size
                    # Get batch of generated ids, padding masks, output masks
                    batch_generated_ids = generated_ids[batch_idx:batch_idx+self.config.batch_size]                 # (B, T)
                    batch_gen_padding_masks = gen_padding_masks[batch_idx:batch_idx+self.config.batch_size]         # (B, T)
                    batch_gen_output_masks = gen_output_masks[batch_idx:batch_idx+self.config.batch_size]           # (B, T)
                    num_output_tokens = batch_gen_output_masks[:, :-1].sum()                                        # scalar
                    # Get old log-probs and values and Advantages
                    batch_old_log_probs = old_log_probs[batch_idx:batch_idx+self.config.batch_size]                 # (B, T-1)
                    batch_advantages = advantages[batch_idx:batch_idx+self.config.batch_size]                       # (B, T-1)
                    batch_returns = returns[batch_idx:batch_idx+self.config.batch_size]                             # (B, T-1)
                    
                    # a. Calculate current policy log-probs, critic values
                    # PRO-TIP --> gradients will flow through this forward pass for PPO model
                    current_log_probs, current_critic_values = self.get_log_probs_and_values(       # (B, T-1), (B, T)
                        self.ppo_model,
                        batch_generated_ids,                                                        # (B, T)
                        batch_gen_padding_masks                                                     # (B, T)
                    )

                    # -----------------------------------------------------------------------------
                    # b. PPO loss
                    # Actor loss
                    prob_ratio = torch.exp(current_log_probs - batch_old_log_probs)                 # (B, T-1)
                    prob_ratio_clipped = torch.clip(                                                # (B, T-1)
                        prob_ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                    )
                    # TODO - log how frequently prob_ratio is getting clipped, 
                    # TODO how frequently loss_actor is clipped ie. zero grad
                    unclipped = prob_ratio *  batch_advantages                                      # (B, T-1)
                    clipped = prob_ratio_clipped * batch_advantages                                 # (B, T-1)
                    loss_actor = - torch.min(unclipped, clipped)                                    # (B, T-1)
                    loss_actor = loss_actor * batch_gen_output_masks[:, :-1]                        # (B, T-1) zero out padding + prompt tokens loss
                    loss_actor = loss_actor.sum() / num_output_tokens                               # scalar - divide by number of output tokens instead of .mean() which would average over padding tokens too

                    # Critic loss
                    value_pred = current_critic_values[:, :-1]                                      # (B, T-1)  ignore last token used for bootstrapping
                    loss_critic = torch.square(batch_returns - value_pred)                           # (B, T-1)
                    loss_critic = loss_critic * batch_gen_output_masks[:, :-1]                      # (B, T-1)  zero out padding + prompt tokens loss
                    loss_critic = loss_critic.sum() / num_output_tokens                             # scalar

                    # PPO loss
                    loss_ppo = loss_actor + self.config.value_loss_coef * loss_critic               # scalar
                    # -----------------------------------------------------------------------------

                    # f. Optimizer step
                    loss_scaled = loss_ppo / self.config.grad_accumulation_steps
                    loss_scaled.backward()                                                             # gradient accumulation

                    if (i+1) % self.config.grad_accumulation_steps == 0:
                        print(f"Loss at {i+1}:", loss_scaled.item())
                        torch.nn.utils.clip_grad_norm_(self.ppo_model.parameters(), max_norm=self.config.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                # end of learning epoch
                print(f"PPO epoch {ppo_epoch+1} learning epoch {learning_epoch+1}/{self.config.learning_epochs} completed")
            # end of output epoch
            print(f"PPO epoch {ppo_epoch+1}/{self.config.ppo_epochs} completed")
        
        # end of train
        print("-"*50)
        print(f"PPO training completed")
        





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
            outputs = reward_model(**encoded).logits            # (b, 3)    Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
            # Convert to probabilities to bound the reward signal strictly between [0, 1]
            probs = torch.softmax(outputs, dim=-1)              # (b, 3)
            positive_score = probs[:, 2]                        # (b,)
            rewards.append(positive_score)
    return torch.cat(rewards)                                   # (B,)


if __name__ == "__main__":
    config = TrainingConfig(
        model_name="Qwen/Qwen3-0.6B",
        reward_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    trainer = PPOTrainer(config)
    # train the model
    trainer.train()



    # # generate responses
    # generated_ids, gen_padding_masks, gen_output_masks, gen_texts_list = trainer.generate_responses()
    # # get log probs and values
    # log_prbs, critic_output = trainer.get_log_probs_and_values(trainer.ppo_model, generated_ids, gen_padding_masks)
    # ref_log_prbs, _ = trainer.get_log_probs_and_values(trainer.ref_model, generated_ids, gen_padding_masks)
    # # get rewards
    # rewards = get_sentiment_rewards(gen_texts_list, trainer.reward_model, trainer.reward_tokenizer, config.reward_batch_size)
    # # compute advantages
    # advantages = trainer.compute_gae_advantages(rewards, log_prbs - ref_log_prbs, critic_output, gen_output_masks)
    # seq_adv = advantages.sum(dim=1)
    # for i in range(len(gen_texts_list)):
    #     print("Score: ", rewards[i].item(), "Seq Adv: ", seq_adv[i].item(), "Text: ", gen_texts_list[i])
    #     print("-"*30)
