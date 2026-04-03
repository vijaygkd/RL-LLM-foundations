import csv
import os
import wandb

class PPOTelemetry:
    def __init__(self, config=None, log_dir="logs", use_wandb=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = []
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project="ppo-transformers", config=config.__dict__ if config else None)
        
        # Current epoch accumulators
        self.current_epoch_metrics = {}
        self.learning_step_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "clip_fraction": []
        }

    def log_generation(self, ppo_epoch, reward, kl_penalty, advantage, returns):
        """Log the static environment variables captured during rollout generation."""
        # Reset epoch metrics
        self.current_epoch_metrics = {
            "ppo_epoch": ppo_epoch,
            "mean_reward": reward.mean().item(),
            "mean_kl_penalty": kl_penalty.mean().item(),
            "mean_advantage": advantage.mean().item(),
            "mean_return": returns.mean().item(),
            "eval_reward": None  # Optional metric evaluated every N steps
        }
        # Reset learning step accumulators
        for k in self.learning_step_metrics:
            self.learning_step_metrics[k].clear()

    def log_learning_step(self, actor_loss, critic_loss, clip_fraction):
        """Append inner-loop metrics to be averaged later."""
        self.learning_step_metrics["actor_loss"].append(actor_loss.item())
        self.learning_step_metrics["critic_loss"].append(critic_loss.item())
        if hasattr(clip_fraction, 'item'):
            self.learning_step_metrics["clip_fraction"].append(clip_fraction.item())
        else:
            self.learning_step_metrics["clip_fraction"].append(clip_fraction)

    def log_eval(self, eval_reward):
        """Log evaluation reward (e.g. from an unseen holdout set)."""
        self.current_epoch_metrics["eval_reward"] = eval_reward

    def finalize_epoch(self):
        """Average inner-loop metrics, flush to history list, and print."""
        for key, values in self.learning_step_metrics.items():
            if values:
                self.current_epoch_metrics[key] = sum(values) / len(values)
            else:
                self.current_epoch_metrics[key] = 0.0
                
        self.metrics_history.append(self.current_epoch_metrics.copy())
        
        # Print summary
        m = self.current_epoch_metrics
        print(f"--- Epoch {m['ppo_epoch'] + 1} Telemetry ---")
        print(f"Reward: {m['mean_reward']:.4f} | KL: {m['mean_kl_penalty']:.4f}")
        print(f"Actor Loss: {m['actor_loss']:.4f} | Critic Loss: {m['critic_loss']:.4f}")
        print(f"Clip Fraction: {m['clip_fraction']:.4f}")
        print("-" * 30)

        if self.use_wandb:
            wandb.log(self.current_epoch_metrics, step=m['ppo_epoch'])

    def save_to_csv(self, filename="ppo_metrics.csv"):
        """Dump the entire history to a CSV file."""
        if not self.metrics_history:
            return
            
        filepath = os.path.join(self.log_dir, filename)
        headers = self.metrics_history[0].keys()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"Telemetry saved to {filepath}")

    def plot(self, save_path="ppo_curves.png"):
        """Plot and save training/evaluation curves."""
        if not self.metrics_history:
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        
        epochs = [m["ppo_epoch"] for m in self.metrics_history]
        rewards = [m["mean_reward"] for m in self.metrics_history]
        actor_losses = [m["actor_loss"] for m in self.metrics_history]
        critic_losses = [m["critic_loss"] for m in self.metrics_history]
        
        # Filter eval rewards (which might be None for some epochs)
        eval_epochs = [m["ppo_epoch"] for m in self.metrics_history if m.get("eval_reward") is not None]
        eval_rewards = [m["eval_reward"] for m in self.metrics_history if m.get("eval_reward") is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Subplot 1: Critic / Actor Losses ---
        ax1.plot(epochs, actor_losses, label="Actor Loss", color="blue", alpha=0.7)
        ax1.set_xlabel("PPO Epoch")
        ax1.set_ylabel("Actor Loss", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.grid(True, alpha=0.3)
        
        # Twin x-axis for Critic Loss since magnitudes will differ drastically
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, critic_losses, label="Critic Target/Loss", color="red", alpha=0.5)
        ax1_twin.set_ylabel("Critic Loss", color="red")
        ax1_twin.tick_params(axis='y', labelcolor="red")
        
        ax1.set_title("PPO Optimization Losses")

        # --- Subplot 2: Rewards ---
        ax2.plot(epochs, rewards, label="Training Reward (Batch)", linewidth=1.5)
        if eval_rewards:
            ax2.plot(eval_epochs, eval_rewards, marker='o', linewidth=2, color='green', label="Eval Reward (Holdout Set)")
            
        ax2.set_xlabel("PPO Epoch")
        ax2.set_ylabel("Mean Scalar Reward")
        ax2.set_title("Reward Tracking")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.log_dir, save_path)
        plt.savefig(filepath, dpi=150)
        print(f"Training curves saved to {filepath}")
        plt.close()

    def finalize_training(self):
        """Cleanup wandb session."""
        if self.use_wandb:
            wandb.finish()
