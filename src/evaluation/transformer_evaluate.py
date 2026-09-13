"""
src/evaluation/transformer_evaluate.py

Metric shape is intentionally identical to
experiments/05_train_ml_selected_features.py so that the ML
ablation table and the transformer ablation table can be
concatenated directly for the paper.
"""

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


@torch.no_grad()
def predict(model, dataloader, device):

    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in dataloader:

        labels = batch.pop("labels")

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )[:, 1]

        preds = torch.argmax(
            outputs.logits,
            dim=-1
        )

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def calculate_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {

        "accuracy":
            accuracy_score(y_true, y_pred),

        "precision":
            precision_score(y_true, y_pred, zero_division=0),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1_score(y_true, y_pred, zero_division=0),

        "roc_auc":
            roc_auc_score(y_true, y_prob),

        "pr_auc":
            average_precision_score(y_true, y_prob),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }