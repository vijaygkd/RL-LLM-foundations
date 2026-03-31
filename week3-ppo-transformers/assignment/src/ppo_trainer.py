import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from ppo_model import PPOModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class TraingConfig:
    model_name: str
    reward_model_name: str
    # dataset
    num_prompts: int = 1000             # ?? Might be better to generate large batch of rollouts in one go 
                                        # to save inference time and reduce GPU wall-time. 
                                        # Or should we do it in batches..?
    num_rollouts_per_prompt: int = 4    # ?? Do we need multiple rollouts per prompt? 
    # training
    no_ppo_steps: int = 1000
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
    2. Load dataset - stanfordnlp/imdb (first 6-8 tokens as prompts only)
    3. Run PPO training loop for E epochs
        a. Generate responses from actor from dataset prompts --> how many rollouts responses per prompt?
        b. Compute sequence-level rewards using static reward model
        for N micro-batches
            c. Calculate timestep (token-level) KL divergence between actor and ref
            d. Compute log-prob, value from PPO model (actor, value heads)
            e. Compute advantage estimates (GAE) at token-level using reward, KL, value
            f. Compute PPO loss = PPO clipped loss + beta * KL divergence penalty 
            g. Update actor and critic: Loss = PPO loss + value loss (MSE)
    """
    def __init__(self, config: TraingConfig):
        print("Using device: ", DEVICE)
        self.config = config
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # PPO model (actor + critic)
        self.ppo_model = PPOModel(config.model_name).to(DEVICE)
        # reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name).to(DEVICE).requires_grad_(False)
        # reward model (frozen)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
        # TODO - can this load the reward model trained in week 2?
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_name).to(DEVICE).requires_grad_(False)

    def generate_responses(self):
        pass

    def compute_rewards(self):
        pass

    def train(self):
        pass




if __name__ == "__main__":
    config = TraingConfig(
        model_name="Qwen/Qwen3-0.6B",
        reward_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
    