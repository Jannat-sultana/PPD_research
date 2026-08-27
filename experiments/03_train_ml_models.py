from pathlib import Path

import pandas as pd

from src.models.models import get_models
from src.models.tuning import (
    get_param_distributions,
    tune_model
)

from src.evaluation.evaluate import (
    evaluate_model,
    get_confusion_matrix,
    get_classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT /
    "src" /
    "processed" /
    "splits" /
    "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT /
    "src" /
    "processed" /
    "splits" /
    "test.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics"
)

TARGET = "Risk"


def main():

    print("=" * 60)
    print("TRADITIONAL ML EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------
    # Load fixed split
    # --------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print("\nTrain:")
    print(train_df.shape)

    print("\nTest:")
    print(test_df.shape)

    # --------------------------------------------------
    # X / y
    # --------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET]
    )

    y_train = train_df[TARGET]

    X_test = test_df.drop(
        columns=[TARGET]
    )

    y_test = test_df[TARGET]

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    models = get_models()

    param_distributions = (
        get_param_distributions()
    )

    results = []

    # --------------------------------------------------
    # Train each model
    # --------------------------------------------------

    for name, model in models.items():

        print("\n")
        print("=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        # ----------------------------------------------
        # Hyperparameter tuning
        # ----------------------------------------------

        search = tune_model(
            model=model,
            param_distribution=(
                param_distributions[name]
            ),
            X_train=X_train,
            y_train=y_train
        )

        print("\nBest parameters:")
        print(search.best_params_)

        print(
            f"\nBest CV F1: "
            f"{search.best_score_:.4f}"
        )

        best_model = (
            search.best_estimator_
        )

        # ----------------------------------------------
        # Test evaluation
        # ----------------------------------------------

        metrics, y_pred, y_score = (
            evaluate_model(
                best_model,
                X_test,
                y_test
            )
        )

        print("\nTest metrics:")

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

        # ----------------------------------------------
        # Confusion matrix
        # ----------------------------------------------

        cm = get_confusion_matrix(
            y_test,
            y_pred
        )

        print("\nConfusion Matrix:")
        print(cm)

        # ----------------------------------------------
        # Classification report
        # ----------------------------------------------

        print("\nClassification Report:")
        print(
            get_classification_report(
                y_test,
                y_pred
            )
        )

        # ----------------------------------------------
        # Save result
        # ----------------------------------------------

        results.append({
            "Model": name,
            **metrics,
            "Best Parameters": str(
                search.best_params_
            ),
            "CV F1": search.best_score_
        })

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        RESULTS_DIR /
        "ml_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("FINAL ML MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nResults saved to:\n{results_path}"
    )


if __name__ == "__main__":
    main()