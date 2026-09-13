import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    results = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            )
    }

    return results


def get_confusion_matrix(
    y_true,
    y_pred
):

    return confusion_matrix(
        y_true,
        y_pred
    )


def get_classification_report(
    y_true,
    y_pred
):

    return classification_report(
        y_true,
        y_pred,
        zero_division=0
    )