from pathlib import Path

import pandas as pd
import yaml


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
# READ FEATURE FILE
# ============================================================

def load_features(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config()

    target = (
        config["data"]["target"]
    )

    # --------------------------------------------------------
    # Load fixed train/test split
    # --------------------------------------------------------

    train = pd.read_csv(
        ROOT /
        config["data"]["train_path"]
    )

    test = pd.read_csv(
        ROOT /
        config["data"]["test_path"]
    )

    print("=" * 70)
    print("PREPARING ML ABLATION DATASETS")
    print("=" * 70)

    print(
        "Train shape:",
        train.shape
    )

    print(
        "Test shape:",
        test.shape
    )

    # --------------------------------------------------------
    # Feature directory
    # --------------------------------------------------------

    feature_dir = (
        ROOT /
        "src" /
        "processed" /
        "selected_features"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        ROOT /
        "src" /
        "processed" /
        "ml_ablation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # All feature-set files
    # --------------------------------------------------------

    feature_files = sorted(
        feature_dir.glob("*.txt")
    )

    if not feature_files:

        raise FileNotFoundError(
            f"No feature files found in "
            f"{feature_dir}"
        )

    summary_rows = []

    # --------------------------------------------------------
    # Process every feature set
    # --------------------------------------------------------

    for feature_file in feature_files:

        feature_set_name = (
            feature_file
            .stem
        )

        features = load_features(
            feature_file
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

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
                f"{feature_set_name}: "
                f"missing features in train: "
                f"{missing_train}"
            )

        if missing_test:

            raise ValueError(
                f"{feature_set_name}: "
                f"missing features in test: "
                f"{missing_test}"
            )

        # ----------------------------------------------------
        # Create datasets
        # ----------------------------------------------------

        train_selected = train[
            features + [target]
        ].copy()

        test_selected = test[
            features + [target]
        ].copy()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        train_output = (
            output_dir /
            f"{feature_set_name}_train.csv"
        )

        test_output = (
            output_dir /
            f"{feature_set_name}_test.csv"
        )

        train_selected.to_csv(
            train_output,
            index=False
        )

        test_selected.to_csv(
            test_output,
            index=False
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_rows.append({
            "feature_set": feature_set_name,
            "n_features": len(features),
            "train_rows": len(train_selected),
            "test_rows": len(test_selected)
        })

        print(
            f"\n{feature_set_name}"
        )

        print(
            f"  Features: {len(features)}"
        )

        print(
            f"  Train:    {train_output}"
        )

        print(
            f"  Test:     {test_output}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir /
        "ablation_dataset_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "Output directory:",
        output_dir
    )

    print(
        "Summary:",
        summary_path
    )


if __name__ == "__main__":

    main()