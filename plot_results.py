import csv

import matplotlib
matplotlib.use("Agg")   # window chhara file-e save korbe
import matplotlib.pyplot as plt

CSV_FILE = "results.csv"
OUT_FILE = "fl_results.png"

rows = list(csv.DictReader(open(CSV_FILE)))
rounds = [int(r["round"]) for r in rows]


def col(name):
    return [float(r[name]) for r in rows]


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1) Accuracy
ax = axes[0]
ax.plot(rounds, col("train_acc"), marker="o", label="Train")
ax.plot(rounds, col("val_acc"), marker="o", label="Validation")
ax.plot(rounds, col("test_acc"), marker="o", label="Test")
ax.set_title("Accuracy per round")
ax.set_xlabel("Round"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

# 2) Loss
ax = axes[1]
ax.plot(rounds, col("train_loss"), marker="o", label="Train")
ax.plot(rounds, col("test_loss"), marker="o", label="Test")
ax.set_title("Loss per round")
ax.set_xlabel("Round"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)

# 3) ECE
ax = axes[2]
ax.plot(rounds, col("test_ece"), marker="o", color="tab:red")
ax.set_title("Test ECE per round (lower = better calibrated)")
ax.set_xlabel("Round"); ax.set_ylabel("ECE"); ax.grid(alpha=0.3)

fig.suptitle("Federated Learning (FedAvg, 4 clients) - EfficientNet-B0 on Kvasir v2")
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=200)
print(f"Saved {OUT_FILE}")
print(f"Final round {rounds[-1]}: test_acc={col('test_acc')[-1]:.4f}, "
      f"test_ece={col('test_ece')[-1]:.4f}")
      