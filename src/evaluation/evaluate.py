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


def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    # -----------------------------------------------
    # Probability / decision score
    # -----------------------------------------------

    if hasattr(model, "predict_proba"):

        y_score = (
            model.predict_proba(X_test)[:, 1]
        )

    elif hasattr(model, "decision_function"):

        y_score = (
            model.decision_function(X_test)
        )

    else:

        y_score = None

    # -----------------------------------------------
    # Metrics
    # -----------------------------------------------

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
    }

    if y_score is not None:

        metrics["ROC-AUC"] = (
            roc_auc_score(
                y_test,
                y_score
            )
        )

        metrics["PR-AUC"] = (
            average_precision_score(
                y_test,
                y_score
            )
        )

    return (
        metrics,
        y_pred,
        y_score
    )


def get_confusion_matrix(
    y_test,
    y_pred
):

    return confusion_matrix(
        y_test,
        y_pred
    )


def get_classification_report(
    y_test,
    y_pred
):

    return classification_report(
        y_test,
        y_pred,
        zero_division=0
    )