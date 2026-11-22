import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data import LABELS


def evaluate(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {"macro_f1": f1_score(y_true, y_pred, average="macro")}
    for i, label in enumerate(LABELS):
        out |= {f"f1_{label}": f1[i], f"precision_{label}": precision[i], f"recall_{label}": recall[i]}
    return out


def per_class(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=LABELS).round(4)


def by_group(df, cols, pred="pred"):
    rows = {key: {"n": len(g), "macro_f1": f1_score(g.label, g[pred], average="macro")} for key, g in df.groupby(cols)}
    return pd.DataFrame(rows).T.astype({"n": int}).round(4)


def plot_confusion(y_true, y_pred, title="", ax=None):
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS)
    share = cm.div(cm.sum(1), axis=0)
    ax = ax or plt.subplots(figsize=(4, 3.5))[1]
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm.iat[i, j]}\n{share.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel="predicted", ylabel="true", title=title)
    return cm
