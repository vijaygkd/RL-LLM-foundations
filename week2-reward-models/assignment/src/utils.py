"""
Utilities for reward model training - telemetry, plotting, etc.
Not part of the core training pipeline.
"""
import numpy as np


class TrainingTelemetry:
    """Logs training loss and eval accuracy, and plots curves at the end."""

    def __init__(self):
        self.losses = []
        self.accuracies = []

    def log_loss(self, loss: float):
        self.losses.append(loss)

    def log_accuracy(self, accuracy: float):
        self.accuracies.append(accuracy)

    def plot(self, save_path: str = "training_curves.png"):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Loss curve ---
        ax1.plot(self.losses, linewidth=0.5, alpha=0.3, label="per-step")
        if len(self.losses) > 50:
            window = min(50, len(self.losses) // 5)
            smoothed = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, window-1+len(smoothed)), smoothed, linewidth=1.5, label=f"smoothed (w={window})")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Accuracy curve ---
        ax2.plot(self.accuracies, marker='o', linewidth=1.5)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="random baseline")
        ax2.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label="target (65%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Eval Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
        plt.close()
