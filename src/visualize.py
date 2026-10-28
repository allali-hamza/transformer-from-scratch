import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


project_dir = Path(__file__).resolve().parent.parent

results_file = project_dir / "results" / "evaluation_results.csv"
figures_dir = project_dir / "results" / "figures"

figures_dir.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(results_file)


train_losses = [
    5.5610, 2.9766, 2.0895, 1.5629, 1.1760,
    0.8832, 0.6526, 0.4811, 0.3556, 0.2720
]

val_losses = [
    3.9130, 2.7823, 2.3409, 2.1188, 2.0329,
    2.0178, 2.0242, 2.0718, 2.1119, 2.2366
]

epochs = range(1, len(train_losses) + 1)


# Training and Validation Loss

plt.figure(figsize=(9, 5))

plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

best_epoch = val_losses.index(min(val_losses)) + 1

plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Best Epoch ({best_epoch})")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(list(epochs))

plt.tight_layout()
plt.savefig(figures_dir / "training_loss.png", dpi=300)
plt.close()


# Evaluation Metrics

total = len(results)
with_argument = results["first_argument_match"].notna()

metrics = {
    "Exact Match": results["exact_match"].mean() * 100,
    "Command": results["command_match"].mean() * 100,
    "First Argument": results.loc[with_argument, "first_argument_match"].mean() * 100,
    "Command + Argument": results.loc[with_argument, "command_and_argument_match"].mean() * 100
}

plt.figure(figsize=(9, 5))

bars = plt.bar(metrics.keys(), metrics.values())

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.ylabel("Accuracy (%)")
plt.title("Transformer Evaluation Metrics")
plt.ylim(0, 100)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(figures_dir / "evaluation_metrics.png", dpi=300)
plt.close()


# Error Distribution

correct_command = results["command_match"].astype(bool)
correct_exact = results["exact_match"].astype(bool)

error_types = {
    "Exact Match": correct_exact.sum(),
    "Correct Command, Wrong Arguments": (correct_command & ~correct_exact).sum(),
    "Wrong Command": (~correct_command).sum()
}

plt.figure(figsize=(9, 5))

bars = plt.bar(error_types.keys(), error_types.values())

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center"
    )

plt.ylabel("Number of Examples")
plt.title("Prediction Outcome Distribution")
plt.xticks(rotation=10)

plt.tight_layout()
plt.savefig(figures_dir / "error_distribution.png", dpi=300)
plt.close()


# Command Accuracy by Command

results["expected_command"] = results["expected"].str.split().str[0]

command_stats = results.groupby("expected_command").agg(
    total=("command_match", "size"),
    correct=("command_match", "sum")
)

command_stats = command_stats[command_stats["total"] >= 10].copy()

command_stats["accuracy"] = (
    command_stats["correct"] / command_stats["total"] * 100
)

command_stats = command_stats.sort_values("total", ascending=False).head(15)

plt.figure(figsize=(10, 6))

bars = plt.barh(command_stats.index, command_stats["accuracy"])

plt.xlabel("Command Accuracy (%)")
plt.ylabel("Linux Command")
plt.title("Command Accuracy by Linux Command")
plt.xlim(0, 100)

plt.gca().invert_yaxis()

for bar in bars:
    plt.text(
        min(bar.get_width() + 1, 95),
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.1f}%",
        va="center"
    )

plt.tight_layout()
plt.savefig(figures_dir / "command_accuracy.png", dpi=300)
plt.close()


print(f"Generated 4 figures in {figures_dir}")