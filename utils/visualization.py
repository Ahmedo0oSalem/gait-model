import matplotlib.pyplot as plt
import numpy as np


def visualize_fold_accuracies(fold_accuracies, save_path=None):
    """
    Visualizes fold-wise accuracy as a bar chart.

    Args:
        fold_accuracies (list or np.array): Accuracy scores for each fold.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    fold_accuracies = np.array(fold_accuracies)
    num_folds = len(fold_accuracies)
    avg_acc = fold_accuracies.mean()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, num_folds + 1), fold_accuracies, color='skyblue', edgecolor='black')
    plt.axhline(avg_acc, color='red', linestyle='--', label=f'Avg Accuracy: {avg_acc:.2f}')

    # Annotate each bar with accuracy
    for bar, acc in zip(bars, fold_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{acc:.2f}",
                 ha='center', va='bottom', fontsize=10)

    plt.title("K-Fold Cross-Validation Accuracies Multiclass")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, num_folds + 1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved accuracy plot to {save_path}")

    plt.show()
