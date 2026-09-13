from pathlib import Path

import pandas as pd
import yaml

from src.feature_selection.selector import (
    rfecv_selection,
    rfe_selection,
    anova_selection,
    build_feature_sets,
    save_feature_sets,
    create_selection_report
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
# MAIN
# ============================================================

def main():

    config = load_config()

    target = (
        config["data"]["target"]
    )

    train_path = (
        ROOT /
        config["data"]["train_path"]
    )

    df = pd.read_csv(
        train_path
    )

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    all_features = X.columns.tolist()

    print("=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    print(
        f"Training shape: {df.shape}"
    )

    print(
        f"Number of predictor features: "
        f"{len(all_features)}"
    )

    # ========================================================
    # RFECV
    # ========================================================

    rfecv_config = (
        config["feature_selection"]["rfecv"]
    )

    rfecv_features = rfecv_selection(
        X,
        y,
        cv=rfecv_config["cv"],
        step=rfecv_config["step"],
        min_features_to_select=(
            rfecv_config[
                "min_features_to_select"
            ]
        ),
        random_state=config["seed"]
    )

    print("\n" + "-" * 70)
    print("RFECV")
    print("-" * 70)

    print(
        f"Number of features: "
        f"{len(rfecv_features)}"
    )

    for feature in rfecv_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # RFE
    # ========================================================

    rfe_config = (
        config["feature_selection"]["rfe"]
    )

    rfe_features = rfe_selection(
        X,
        y,
        n_features=rfe_config["n_features"],
        random_state=config["seed"]
    )

    print("\n" + "-" * 70)
    print("RFE")
    print("-" * 70)

    print(
        f"Number of features: "
        f"{len(rfe_features)}"
    )

    for feature in rfe_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # ANOVA
    # ========================================================

    anova_config = (
        config["feature_selection"]["anova"]
    )

    anova_features = anova_selection(
        X,
        y,
        k=anova_config["k"]
    )

    print("\n" + "-" * 70)
    print("ANOVA")
    print("-" * 70)

    print(
        f"Number of features: "
        f"{len(anova_features)}"
    )

    for feature in anova_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # BUILD ALL ABLATION SETS
    # ========================================================

    feature_sets = build_feature_sets(
        all_features=all_features,
        rfecv_features=rfecv_features,
        rfe_features=rfe_features,
        anova_features=anova_features
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE SET SUMMARY")
    print("=" * 70)

    for name, features in feature_sets.items():

        print(
            f"{name:<30} "
            f"{len(features):>3} features"
        )

    # ========================================================
    # SAVE FEATURE SETS
    # ========================================================

    feature_output_dir = (
        ROOT /
        "src" /
        "processed" /
        "selected_features"
    )

    save_feature_sets(
        feature_sets,
        feature_output_dir
    )

    # ========================================================
    # SAVE SELECTION REPORT
    # ========================================================

    report = create_selection_report(
        all_features=all_features,
        rfecv_features=rfecv_features,
        rfe_features=rfe_features,
        anova_features=anova_features
    )

    report_dir = (
        ROOT /
        "results" /
        "feature_selection"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        report_dir /
        "feature_selection_summary.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    # ========================================================
    # SAVE FEATURE-SET SUMMARY
    # ========================================================

    summary = pd.DataFrame({
        "feature_set": list(
            feature_sets.keys()
        ),
        "n_features": [
            len(features)
            for features
            in feature_sets.values()
        ]
    })

    summary_path = (
        report_dir /
        "feature_set_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        "Feature files:",
        feature_output_dir
    )

    print(
        "Selection report:",
        report_path
    )

    print(
        "Feature-set summary:",
        summary_path
    )


if __name__ == "__main__":
    main()