from pathlib import Path

import pandas as pd
import yaml

from src.feature_selection.selector import (
    rfecv_selection,
    rfe_selection,
    anova_selection,
    save_features
)


ROOT = Path(
    __file__
).resolve().parents[1]


CONFIG = ROOT / "configs" / "base.yaml"


def load_config():

    with open(CONFIG) as f:

        return yaml.safe_load(f)


def main():

    config = load_config()

    target = config["data"]["target"]

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

    print("=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)

    print(
        "Training shape:",
        df.shape
    )

    # --------------------------------------------------
    # RFECV
    # --------------------------------------------------

    rfecv_features = rfecv_selection(
        X,
        y,
        cv=config["feature_selection"]
        ["rfecv"]["cv"],
        step=config["feature_selection"]
        ["rfecv"]["step"],
        min_features_to_select=config
        ["feature_selection"]
        ["rfecv"]
        ["min_features_to_select"]
    )

    print(
        "\nRFECV features:"
    )

    print(
        rfecv_features
    )

    # --------------------------------------------------
    # RFE
    # --------------------------------------------------

    rfe_features = rfe_selection(
        X,
        y,
        config["feature_selection"]
        ["rfe"]
        ["n_features"]
    )

    print(
        "\nRFE features:"
    )

    print(
        rfe_features
    )

    # --------------------------------------------------
    # ANOVA
    # --------------------------------------------------

    anova_features = anova_selection(
        X,
        y,
        config["feature_selection"]
        ["anova"]
        ["k"]
    )

    print(
        "\nANOVA features:"
    )

    print(
        anova_features
    )

    # --------------------------------------------------
    # Consensus
    # --------------------------------------------------

    feature_sets = {
        "RFECV": set(rfecv_features),
        "RFE": set(rfe_features),
        "ANOVA": set(anova_features)
    }

    consensus = (
        feature_sets["RFECV"]
        &
        feature_sets["RFE"]
        &
        feature_sets["ANOVA"]
    )

    # If strict intersection is too small,
    # use union frequency ranking.

    if len(consensus) < 3:

        frequency = {}

        for features in feature_sets.values():

            for feature in features:

                frequency[feature] = (
                    frequency.get(
                        feature,
                        0
                    ) + 1
                )

        selected_features = [
            feature
            for feature, count
            in sorted(
                frequency.items(),
                key=lambda x: (
                    -x[1],
                    x[0]
                )
            )
            if count >= 2
        ]

    else:

        selected_features = sorted(
            consensus
        )

    print(
        "\nFINAL SELECTED FEATURES:"
    )

    for feature in selected_features:

        print(
            feature
        )

    output_path = (
        ROOT /
        config["data"]
        ["selected_features_path"]
    )

    save_features(
        selected_features,
        output_path
    )

    # --------------------------------------------------
    # Save selection report
    # --------------------------------------------------

    report = pd.DataFrame({

        "feature": list(
            set(
                rfecv_features
                +
                rfe_features
                +
                anova_features
            )
        )

    })

    report[
        "RFECV"
    ] = report[
        "feature"
    ].isin(
        rfecv_features
    )

    report[
        "RFE"
    ] = report[
        "feature"
    ].isin(
        rfe_features
    )

    report[
        "ANOVA"
    ] = report[
        "feature"
    ].isin(
        anova_features
    )

    report[
        "frequency"
    ] = (
        report[
            [
                "RFECV",
                "RFE",
                "ANOVA"
            ]
        ].sum(axis=1)
    )

    report = report.sort_values(
        "frequency",
        ascending=False
    )

    report_path = (
        ROOT /
        "results" /
        "feature_selection.csv"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        report_path,
        index=False
    )


if __name__ == "__main__":

    main()