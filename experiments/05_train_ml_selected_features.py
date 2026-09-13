from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.base import clone

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CONFIG = (
    ROOT /
    "configs" /
    "base.yaml"
)


# ============================================================
# CONFIG
# ============================================================

def load_config():

    with open(
        CONFIG,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# MODELS
# ============================================================

def get_models(
    random_state=42
):

    models = {

        "LogisticRegression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),

        "ExtraTrees":
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=random_state
            ),

        "SVM":
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            )
    }

    return models


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return features


# ============================================================
# PROBABILITY PREDICTION
# ============================================================

def get_probabilities(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        return model.predict_proba(
            X
        )[:, 1]

    if hasattr(
        model,
        "decision_function"
    ):

        return model.decision_function(
            X
        )

    raise ValueError(
        "Model does not support "
        "probability or decision predictions."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {

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

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

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
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config()

    target = (
        config["data"]["target"]
    )

    random_state = (
        config["seed"]
    )

    # ========================================================
    # LOAD FIXED SPLIT
    # ========================================================

    train_path = (
        ROOT /
        config["data"]["train_path"]
    )

    test_path = (
        ROOT /
        config["data"]["test_path"]
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    print("=" * 70)
    print("ML ABLATION STUDY")
    print("=" * 70)

    print(
        f"Training samples: "
        f"{len(train)}"
    )

    print(
        f"Test samples: "
        f"{len(test)}"
    )

    print(
        f"Target: {target}"
    )

    # ========================================================
    # FEATURE SETS
    # ========================================================

    feature_dir = (
        ROOT /
        "src" /
        "processed" /
        "selected_features"
    )

    feature_set_names = [
        "all",
        "rfecv",
        "lasso",
        "mrmr",
        "combined"
    ]

    # ========================================================
    # MODELS
    # ========================================================

    models = get_models(
        random_state=random_state
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    results_dir = (
        ROOT /
        "results" /
        "ml"
    )

    predictions_dir = (
        results_dir /
        "predictions"
    )

    models_dir = (
        results_dir /
        "models"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # RESULTS
    # ========================================================

    all_results = []

    # ========================================================
    # FEATURE SET LOOP
    # ========================================================

    for feature_set_name in feature_set_names:

        feature_file = (
            feature_dir /
            f"{feature_set_name}.txt"
        )

        if not feature_file.exists():

            raise FileNotFoundError(
                f"Feature file not found:\n"
                f"{feature_file}\n\n"
                f"Run:\n"
                f"python -m "
                f"experiments.04_feature_selection"
            )

        features = load_features(
            feature_file
        )

        print("\n" + "=" * 70)
        print(
            f"FEATURE SET: "
            f"{feature_set_name.upper()}"
        )
        print("=" * 70)

        print(
            f"Number of features: "
            f"{len(features)}"
        )

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        missing_train = [
            feature
            for feature in features
            if feature not in train.columns
        ]

        missing_test = [
            feature
            for feature in features
            if feature not in test.columns
        ]

        if missing_train:

            raise ValueError(
                f"Missing train features: "
                f"{missing_train}"
            )

        if missing_test:

            raise ValueError(
                f"Missing test features: "
                f"{missing_test}"
            )

        # ====================================================
        # X / Y
        # ====================================================

        X_train = train[
            features
        ].copy()

        y_train = train[
            target
        ].copy()

        X_test = test[
            features
        ].copy()

        y_test = test[
            target
        ].copy()

        # ====================================================
        # MODEL LOOP
        # ====================================================

        for model_name, base_model in models.items():

            print(
                f"\nTraining "
                f"{model_name}..."
            )

            model = clone(
                base_model
            )

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )

            y_prob = get_probabilities(
                model,
                X_test
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            metrics = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob
            )

            result = {

                "feature_set":
                    feature_set_name.upper(),

                "n_features":
                    len(features),

                "model":
                    model_name,

                **metrics
            }

            all_results.append(
                result
            )

            # ------------------------------------------------
            # PRINT RESULTS
            # ------------------------------------------------

            print(
                f"  Accuracy:    "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Precision:   "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"  Sensitivity: "
                f"{metrics['sensitivity']:.4f}"
            )

            print(
                f"  Specificity: "
                f"{metrics['specificity']:.4f}"
            )

            print(
                f"  F1:          "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"  ROC-AUC:     "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"  PR-AUC:      "
                f"{metrics['pr_auc']:.4f}"
            )

            # =================================================
            # SAVE TEST PREDICTIONS
            # =================================================

            predictions = pd.DataFrame({

                "y_true":
                    y_test.to_numpy(),

                "y_pred":
                    y_pred,

                "y_probability":
                    y_prob
            })

            prediction_path = (
                predictions_dir /
                f"{feature_set_name}_"
                f"{model_name}_"
                f"predictions.csv"
            )

            predictions.to_csv(
                prediction_path,
                index=False
            )

            # =================================================
            # SAVE MODEL
            # =================================================

            model_path = (
                models_dir /
                f"{feature_set_name}_"
                f"{model_name}.joblib"
            )

            joblib.dump(
                model,
                model_path
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SAVE COMPLETE RESULTS
    # ========================================================

    results_path = (
        results_dir /
        "ml_ablation_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # SAVE CLEAN SUMMARY
    # ========================================================

    summary_columns = [

        "feature_set",

        "n_features",

        "model",

        "accuracy",

        "precision",

        "sensitivity",

        "specificity",

        "f1",

        "roc_auc",

        "pr_auc"
    ]

    summary = results[
        summary_columns
    ].copy()

    summary_path = (
        results_dir /
        "ml_ablation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # BEST MODEL BY F1
    # ========================================================

    best_f1 = (
        results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST TEST-SET RESULT")
    print("=" * 70)

    print(
        f"Feature set: "
        f"{best_f1['feature_set']}"
    )

    print(
        f"Model:       "
        f"{best_f1['model']}"
    )

    print(
        f"Features:    "
        f"{best_f1['n_features']}"
    )

    print(
        f"F1:          "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"ROC-AUC:     "
        f"{best_f1['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:      "
        f"{best_f1['pr_auc']:.4f}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ML ABLATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Results:\n{results_path}"
    )

    print(
        f"\nSummary:\n{summary_path}"
    )

    print(
        f"\nPredictions:\n{predictions_dir}"
    )

    print(
        f"\nModels:\n{models_dir}"
    )


if __name__ == "__main__":

    main()