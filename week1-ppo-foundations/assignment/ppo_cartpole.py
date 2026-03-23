import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

from utils import plot_learning_curve, save_run_metrics

# 1. The Hyperparameters (Start with these)
LR = 3e-4
DISCOUNT = 0.99
GAE_PARAM = 0.95
EPS_CLIP = 0.2
VF_COEF = 1
ENTROPY_COEF = 0.01
EPOCHS = 10             # How many times to reuse the dataset
BATCH_SIZE = 64         # Mini-batch size
ROLLOUT_LEN = 2048      # How much data to collect before updating
NUM_UPDATES = 50        # No. of policy update loops
# total timesteps = rollout len * num_updates = 100K (enough for cart-pole)
EVAL_RUNS_PER_UPDATE = 10
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# 2. The Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        print(f"Using device: {DEVICE}")

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        ).to(DEVICE)   # output logits for actions

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(DEVICE)   # output value score
        

    def forward(self, state):
        action, log_prob = self.get_action(state)
        value = self.get_value(state)
        return action, log_prob, value

    def get_action(self, state):
        # 1. Pass state through Actor
        logits = self.actor(state) # (,2)
        # 2. Create a distribution (Categorical)
        # https://docs.pytorch.org/docs/stable/distributions.html#categorical
        # Why Categorical? It is not possible to back prop through random samples.
        # REINFORCE is a trick for creating surrogate function that can be backproped.
        # This is commonly used for policy-gradient methods in RL.
        dist = torch.distributions.Categorical(logits=logits)
        # 3. Sample an action
        action = dist.sample()  # sampled index (,1)
        log_prob = dist.log_prob(action)    # (, 1)
        # 4. Return action and its log_prob
        return action, log_prob

    def get_log_prob_entropy(self, state, action):
        logits = self.actor(state)  # (, 2)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        return log_prob, dist.entropy()
    
    def get_value(self, state):
        value = self.critic(state)
        return value

# 3. The PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = [] # Store your rollouts here

    def collect_rollout(self):
        """
        Use "old" / "current" policy to sample fresh data for training. 
        On-policy learning.
        """
        self.buffer = []
        state, info = self.env.reset()
        episode_reward = 0
        for i in range(ROLLOUT_LEN):
            with torch.no_grad():
                # Note: training data is constant, even though generated with policy. 
                # So no_grad applied. (hard bug to catch)
                action, log_prob, value = self.policy(torch.Tensor(state).to(DEVICE))
            new_state, reward, done, truncated, info = self.env.step(action.item())
            self.buffer.append(
                (state, new_state, action.detach(), log_prob.detach(), value.detach(), reward, done)
            )
            state = new_state
            episode_reward += reward
            if done or truncated:
                state, info = self.env.reset()
                episode_reward = 0
    
    def compute_gae(self):
        # Calculate GAE calculation for all timestep in rollout
        # Lookahead till end of episode or end of buffer recursively
        gae_buffer = []                 # gae for last timestep in buffer, since we can't see future
        gae = 0
        value_next = 0
        # init last timestamp if not done
        last_done = self.buffer[-1][6]
        if not last_done:
            # end of buffer is not done state, so we "assume" future values using value model
            next_state = self.buffer[-1][1]
            value_next = self.policy.get_value(torch.Tensor(next_state).to(DEVICE)).item()    

        for i in reversed(range(len(self.buffer))): # start reversed
            b = self.buffer[i]
            value, reward, done = b[4], b[5], b[6]
            if done:  # end of episode -- reset
                value_next = 0
                gae = 0

            delta = (reward + DISCOUNT * value_next) - value    # TD error
            gae = delta + (DISCOUNT * GAE_PARAM * gae)  # recursive formula
            gae_buffer.append(gae)
            value_next = value

        gae_buffer.reverse()
        return gae_buffer

    def _iter_minibatches(self, dataset, batch_size):
        total_size = dataset["states"].shape[0]
        indices = torch.randperm(total_size, device=DEVICE)
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_idx = indices[start:end]
            yield {key: value[batch_idx] for key, value in dataset.items()}

    def _build_dataset(self):
        advantages = self.compute_gae()
        return {
            "states": torch.tensor(
                np.array([b[0] for b in self.buffer]), dtype=torch.float32, device=DEVICE
            ),
            "actions": torch.stack([b[2] for b in self.buffer]).to(DEVICE),
            "log_prob": torch.stack([b[3] for b in self.buffer]).to(DEVICE).detach(),
            "values": torch.stack([b[4] for b in self.buffer]).to(DEVICE).detach(),
            "advantage": torch.tensor(
                [a.item() if torch.is_tensor(a) else a for a in advantages],
                dtype=torch.float32,
                device=DEVICE,
            ),
        }

    def update(self):
        """
        Learning loop / backprop
        """
        dataset = self._build_dataset()     # rollouts, GAE, rewards, etc.

        # TRAINING LOOP
        for i in range(EPOCHS):
            for batch in self._iter_minibatches(dataset, BATCH_SIZE):
                self.optimizer.zero_grad()
                adv = batch['advantage']
                states = batch['states']
                actions = batch['actions']
                log_prob_old = batch['log_prob']              # from old policy rollouts
                values_old = batch['values'].flatten() # (b)

                # on-policy update (sample log_probs, values from latest policy for past actions)
                # gradients flow through these tensors from new policy
                log_prob_new, entropy = self.policy.get_log_prob_entropy(states, actions) # (, 1)
                values_new = self.policy.get_value(states).flatten()  # (b)
                
                # NOTE: Critical (value model) target value.
                # Reward: rewards received at current timestep. (aka. actual outcome)
                # Return: total future discounted rewards. (aka. target value for critic)
                # Adv = actual outcome (aka. target value) - expected baseline (aka. old value)
                # Adv = (received rewards + discounted future returns) - expected baseline
                # So the target value for critic training is 
                # values_target = values_old (baseline) + adv (discoutned sum of future TD errors)
                # Note - Alternative, sub-optimal way is to use sum actual future rewards (non-discounted), but it has high variance.
                
                values_target = values_old + adv     # ** Target for critic model. (b)

                # PPO
                norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(
                    log_prob_new - log_prob_old
                )
                unclipped = ratio * norm_adv
                clipped = torch.clip(
                    ratio, 1 - EPS_CLIP, 1 + EPS_CLIP
                ) * norm_adv
                # clipped surrogate loss -- PPO loss
                actor_loss = - torch.min(unclipped, clipped).mean()
                # MSE
                critic_loss = torch.square(values_new - values_target).mean()
                # combined losses with entropy bonus to allow exploration
                loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) # gradient clipping
                self.optimizer.step()


# 4. Main Training Loop
def train(mode=None, run_name=""):
    env = gym.make("CartPole-v1", render_mode=mode)
    agent = PPOAgent(env)
    eval_avg_history = []
    eval_rewards_history = []
    update_idx = []

    # train loop
    for i in trange(NUM_UPDATES, desc="Update loop", unit="epoch"): # Number of updates
        agent.collect_rollout()
        agent.update()

        # evalute agent and log avg reward per episode
        eval_rewards = evaluate(agent, epochs=EVAL_RUNS_PER_UPDATE)
        eval_avg = float(np.mean(eval_rewards))
        eval_rewards_history.append(eval_rewards)
        eval_avg_history.append(eval_avg)
        update_idx.append(i + 1)
        print(f"Update {i + 1}: avg reward over {EVAL_RUNS_PER_UPDATE} eval runs = {eval_avg:.2f}")

    # logging
    plot_learning_curve(update_idx, eval_avg_history, EVAL_RUNS_PER_UPDATE, run_name=run_name)
    save_run_metrics(run_name, update_idx, eval_avg_history, eval_rewards_history)

    return agent


def evaluate(agent, mode=None, epochs=10):
    env = gym.make("CartPole-v1", render_mode=mode)
    episode_rewards = []
    for i in range(epochs):
        state, info = env.reset()
        episode_reward = 0
        for j in range(ROLLOUT_LEN):
            with torch.no_grad():
                action, log_prob, value = agent.policy(torch.Tensor(state).to(DEVICE))
            state, reward, done, truncated, info = env.step(action.item())
            episode_reward += reward
            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    env.close()
    return episode_rewards

if __name__ == "__main__":
    exp_name = "PPO full"
    print(f"Train PPO agent -> {exp_name}")
    agent = train(mode=None, run_name=exp_name)

    print(f"PPO Final agent eval -> {exp_name}")
    results = evaluate(agent, mode=None, epochs=100)
    print(f"Average eval reward over 100 episodes: {sum(results) / len(results)}")
