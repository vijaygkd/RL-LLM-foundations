"""
Utilities for reward model training - telemetry, plotting, etc.
Not part of the core training pipeline.
"""
import numpy as np


class TrainingTelemetry:
    """Logs training loss and eval accuracy, and plots curves at the end."""

    def __init__(self):
        self.loss_steps = []
        self.losses = []
        self.acc_steps = []
        self.accuracies = []
        self.final_accuracy = None

    def log_loss(self, step: int, loss: float):
        self.loss_steps.append(step)
        self.losses.append(loss)

    def log_accuracy(self, step: int, accuracy: float):
        self.acc_steps.append(step)
        self.accuracies.append(accuracy)

    def set_final_benchmark(self, accuracy: float):
        self.final_accuracy = accuracy

    def plot(self, save_path: str = "training_curves.png"):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss curve ---
        if self.loss_steps:
            ax1.plot(self.loss_steps, self.losses, linewidth=0.5, alpha=0.3, label="per-log-step")
            if len(self.losses) > 10:
                window = min(20, len(self.losses) // 3)
                smoothed = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                ax1.plot(self.loss_steps[window-1:], smoothed, linewidth=1.5, label=f"smoothed (w={window})")
        
        ax1.set_xlabel("Gradient Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Accuracy curve ---
        if self.acc_steps:
            ax2.plot(self.acc_steps, self.accuracies, marker='o', linewidth=1.5, label="Interim Eval (subset)")
            
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random Baseline (50%)")
        ax2.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label="Assignment Target (65%)")
        
        if self.final_accuracy is not None:
             ax2.axhline(y=self.final_accuracy, color='purple', linestyle='-', linewidth=2, label=f"Final Test Accuracy ({self.final_accuracy:.1%})")

        ax2.set_xlabel("Gradient Step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Evaluation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
        plt.close()
