from contextlib import contextmanager
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class PPOModel(nn.Module):
    """
    Wrapper for pretrained causal LM model for PPO training.
    Actor: Pretrained causal LM model
    Critic: Pretrained causal LM model with linear scalar head.
    
    Note: The Actor and Critic will share the same transformer backbone weights.
    This is the modern approach to PPO with LLMs, although the original RLHF paper
    used separate models for actor and critic.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.actor = AutoModelForCausalLM.from_pretrained(model_name)
        self.lm_head = self.actor.lm_head
        self.value_head = nn.Linear(self.lm_head.in_features, 1).to(dtype=self.actor.dtype)

    def forward(self, input_ids, attention_mask):
        """
        Returns LM logits and value estimates
        """
        with capture_inputs(self.actor.lm_head) as act:
            lm_output = self.actor(input_ids, attention_mask)
            lm_input = act["input"][0]

        print("lm_input.shape: ", lm_input.shape)
        value_output = self.value_head(lm_input)
        return lm_output, value_output


@contextmanager
def capture_inputs(module):
    """
    Context manager to capture the inputs to a module.
    """
    activation = {}
    hook = module.register_forward_pre_hook(
        lambda m, i: activation.update({"input": i})
    )
    try:
        yield activation
    finally:
        hook.remove()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "Qwen/Qwen3-0.6B"
    model = PPOModel(model_name).to(DEVICE)
    print(model.actor)
    input_ids = torch.randint(0, 1000, (1, 10)).to(DEVICE)
    attention_mask = torch.ones((1, 10)).to(DEVICE)
    lm_output, value_output = model(input_ids, attention_mask)
    print("lm_output.logits.shape: ", lm_output.logits.shape)
    print("value_output.shape: ", value_output.shape)
    print("value_output: ", value_output)
    # generate text from actor
    tok = AutoTokenizer.from_pretrained(model_name)
    inputs_text = "The movie was"
    input_ids = tok(inputs_text, return_tensors="pt").input_ids.to(DEVICE)
    generated_text = model.actor.generate(input_ids, do_sample=True, max_new_tokens=16)
    print("generated_text: ", tok.decode(generated_text[0]))
    
"""
Model list:
openai-community/gpt2-xl (2B)
Qwen/Qwen2.5-0.5B-Instruct (0.5B)
Qwen/Qwen3-0.6B (0.6B)
Qwen/Qwen3.5-0.8B-Base (0.8B)
Qwen/Qwen3.5-2B-Base (2B)
"""