from pathlib import Path

import pandas as pd

from sklearn.feature_selection import (
    RFECV,
    RFE,
    SelectKBest,
    f_classif
)

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# GENERAL
# ============================================================

def get_feature_columns(df, target):
    """
    Return all predictor columns excluding the target.
    """
    return [
        column
        for column in df.columns
        if column != target
    ]


# ============================================================
# RFECV
# ============================================================

def rfecv_selection(
    X,
    y,
    cv=5,
    step=1,
    min_features_to_select=5,
    random_state=42
):
    """
    Feature selection using RFECV with Random Forest.
    """

    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="f1",
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )

    selector.fit(X, y)

    features = X.columns[
        selector.support_
    ].tolist()

    return features


# ============================================================
# RFE
# ============================================================

def rfe_selection(
    X,
    y,
    n_features,
    random_state=42
):
    """
    Feature selection using RFE with Random Forest.
    """

    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1
    )

    selector.fit(X, y)

    features = X.columns[
        selector.support_
    ].tolist()

    return features


# ============================================================
# ANOVA
# ============================================================

def anova_selection(
    X,
    y,
    k
):
    """
    Feature selection using ANOVA F-test.
    """

    k = min(k, X.shape[1])

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    selector.fit(X, y)

    features = X.columns[
        selector.get_support()
    ].tolist()

    return features


# ============================================================
# FEATURE-SET COMBINATIONS
# ============================================================

def build_feature_sets(
    all_features,
    rfecv_features,
    rfe_features,
    anova_features
):
    """
    Build feature sets for ablation studies.

    Returns a dictionary where each key is an experimental
    condition and each value is a list of features.
    """

    all_set = set(all_features)
    rfecv_set = set(rfecv_features)
    rfe_set = set(rfe_features)
    anova_set = set(anova_features)

    # --------------------------------------------------------
    # Feature combinations
    # --------------------------------------------------------

    feature_sets = {

        # All original features
        "ALL": all_set,

        # Individual methods
        "RFECV": rfecv_set,
        "RFE": rfe_set,
        "ANOVA": anova_set,

        # Pairwise UNION
        "RFECV_ANOVA_UNION": (
            rfecv_set | anova_set
        ),

        "RFECV_RFE_UNION": (
            rfecv_set | rfe_set
        ),

        "RFE_ANOVA_UNION": (
            rfe_set | anova_set
        ),

        # Pairwise INTERSECTION
        "RFECV_ANOVA_INTERSECTION": (
            rfecv_set & anova_set
        ),

        "RFECV_RFE_INTERSECTION": (
            rfecv_set & rfe_set
        ),

        "RFE_ANOVA_INTERSECTION": (
            rfe_set & anova_set
        ),

        # Union of all three
        "ALL_METHODS_UNION": (
            rfecv_set |
            rfe_set |
            anova_set
        ),

        # Features selected by at least 2 methods
        "CONSENSUS_2PLUS": {
            feature
            for feature in all_set
            if (
                int(feature in rfecv_set) +
                int(feature in rfe_set) +
                int(feature in anova_set)
            ) >= 2
        },

        # Features selected by all 3 methods
        "CONSENSUS_ALL": (
            rfecv_set &
            rfe_set &
            anova_set
        )
    }

    # --------------------------------------------------------
    # Convert sets to sorted lists
    # --------------------------------------------------------

    feature_sets = {
        name: sorted(features)
        for name, features in feature_sets.items()
    }

    return feature_sets


# ============================================================
# SAVE FEATURE LIST
# ============================================================

def save_features(
    features,
    path
):
    """
    Save one feature per line.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:

        for feature in features:

            f.write(
                feature + "\n"
            )


# ============================================================
# SAVE ALL FEATURE SETS
# ============================================================

def save_feature_sets(
    feature_sets,
    output_dir
):
    """
    Save every feature set as a separate TXT file.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, features in feature_sets.items():

        path = (
            output_dir /
            f"{name.lower()}.txt"
        )

        save_features(
            features,
            path
        )


# ============================================================
# FEATURE SELECTION REPORT
# ============================================================

def create_selection_report(
    all_features,
    rfecv_features,
    rfe_features,
    anova_features
):
    """
    Create a feature-level selection report.
    """

    report = pd.DataFrame({
        "feature": all_features
    })

    report["RFECV"] = (
        report["feature"]
        .isin(rfecv_features)
    )

    report["RFE"] = (
        report["feature"]
        .isin(rfe_features)
    )

    report["ANOVA"] = (
        report["feature"]
        .isin(anova_features)
    )

    report["frequency"] = (
        report[
            [
                "RFECV",
                "RFE",
                "ANOVA"
            ]
        ].sum(axis=1)
    )

    report["selection_group"] = (
        report["frequency"]
        .map({
            3: "All 3 methods",
            2: "2 methods",
            1: "1 method",
            0: "None"
        })
    )

    report = report.sort_values(
        by=[
            "frequency",
            "feature"
        ],
        ascending=[
            False,
            True
        ]
    )

    return report