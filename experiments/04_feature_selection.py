from pathlib import Path

import pandas as pd
import yaml

from src.feature_selection.selector import (
    rfecv_selection,
    lasso_selection,
    mrmr_selection_discrete,
    build_ablation_feature_sets,
    save_ablation_feature_sets,
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
        f"Number of predictors: "
        f"{len(all_features)}"
    )

    # ========================================================
    # RFECV
    # ========================================================

    rfecv_config = (
        config[
            "feature_selection"
        ][
            "rfecv"
        ]
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

    print("\n" + "=" * 70)
    print("RFECV")
    print("=" * 70)

    print(
        f"Number of features: "
        f"{len(rfecv_features)}"
    )

    for feature in rfecv_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # LASSO
    # ========================================================

    lasso_config = (
        config[
            "feature_selection"
        ][
            "lasso"
        ]
    )

    lasso_features = lasso_selection(
        X,
        y,
        C=lasso_config["C"],
        max_iter=lasso_config["max_iter"],
        random_state=config["seed"]
    )

    print("\n" + "=" * 70)
    print("LASSO")
    print("=" * 70)

    print(
        f"Number of features: "
        f"{len(lasso_features)}"
    )

    for feature in lasso_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # mRMR
    # ========================================================

    mrmr_config = (
        config[
            "feature_selection"
        ][
            "mrmr"
        ]
    )

    mrmr_features = mrmr_selection_discrete(
        X,
        y,
        k=mrmr_config["k"],
        random_state=config["seed"]
    )

    print("\n" + "=" * 70)
    print("mRMR")
    print("=" * 70)

    print(
        f"Number of features: "
        f"{len(mrmr_features)}"
    )

    for feature in mrmr_features:
        print(
            f"  - {feature}"
        )

    # ========================================================
    # BUILD ABLATION SETS
    # ========================================================

    feature_sets = (
        build_ablation_feature_sets(
            all_features=all_features,
            rfecv_features=rfecv_features,
            lasso_features=lasso_features,
            mrmr_features=mrmr_features
        )
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ABLATION FEATURE SET SUMMARY")
    print("=" * 70)

    for name, features in feature_sets.items():

        print(
            f"{name:<15} "
            f"{len(features):>3} features"
        )

    # ========================================================
    # OVERLAP INFORMATION
    # ========================================================

    rfecv_set = set(
        rfecv_features
    )

    lasso_set = set(
        lasso_features
    )

    mrmr_set = set(
        mrmr_features
    )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION OVERLAP")
    print("=" * 70)

    print(
        "RFECV ∩ LASSO:",
        len(
            rfecv_set &
            lasso_set
        )
    )

    print(
        "RFECV ∩ mRMR:",
        len(
            rfecv_set &
            mrmr_set
        )
    )

    print(
        "LASSO ∩ mRMR:",
        len(
            lasso_set &
            mrmr_set
        )
    )

    print(
        "All three:",
        len(
            rfecv_set &
            lasso_set &
            mrmr_set
        )
    )

    # ========================================================
    # SAVE FEATURE FILES
    # ========================================================

    feature_output_dir = (
        ROOT /
        "src" /
        "processed" /
        "selected_features"
    )

    save_ablation_feature_sets(
        feature_sets,
        feature_output_dir
    )

    # ========================================================
    # SAVE FEATURE REPORT
    # ========================================================

    report = create_selection_report(
        all_features=all_features,
        rfecv_features=rfecv_features,
        lasso_features=lasso_features,
        mrmr_features=mrmr_features
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
    # SAVE FEATURE SET SUMMARY
    # ========================================================

    summary = pd.DataFrame({

        "feature_set":
            list(
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
    # DONE
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)

    print(
        "\nFeature files saved to:"
    )

    print(
        feature_output_dir
    )

    print(
        "\nFeature selection report:"
    )

    print(
        report_path
    )

    print(
        "\nFeature-set summary:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":

    main()