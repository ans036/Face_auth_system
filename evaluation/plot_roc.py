import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_true = np.load("evaluation/output/y_true.npy")
y_scores = np.load("evaluation/output/y_scores.npy")

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Acceptance Rate")
plt.ylabel("True Acceptance Rate")
plt.title("ROC Curve – ArcFace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
