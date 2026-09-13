from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    RFECV,
    mutual_info_classif
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler


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
# RANDOM FOREST ESTIMATOR
# ============================================================

def get_rf_estimator(random_state=42):
    """
    Random Forest estimator used by RFECV.
    """

    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )


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
    Recursive Feature Elimination with Cross-Validation.

    Feature selection is performed only on the supplied
    training data.
    """

    estimator = get_rf_estimator(
        random_state=random_state
    )

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="f1",
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )

    selector.fit(
        X,
        y
    )

    features = X.columns[
        selector.support_
    ].tolist()

    return features


# ============================================================
# LASSO FEATURE SELECTION
# ============================================================

def lasso_selection(
    X,
    y,
    C=0.1,
    max_iter=5000,
    random_state=42
):
    """
    LASSO-based feature selection using L1-regularized
    logistic regression.

    Features with non-zero coefficients are retained.

    StandardScaler is used because L1 regularization is
    sensitive to feature scale.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state
    )

    model.fit(
        X_scaled,
        y
    )

    coefficients = np.abs(
        model.coef_[0]
    )

    selected_mask = (
        coefficients > 1e-8
    )

    features = X.columns[
        selected_mask
    ].tolist()

    return features


# ============================================================
# mRMR FEATURE SELECTION
# ============================================================

def mrmr_selection(
    X,
    y,
    k=10,
    random_state=42
):
    """
    Approximate mRMR feature selection.

    Maximum Relevance:
        Mutual information between each feature and target.

    Minimum Redundancy:
        Mutual information between candidate features
        and already selected features.

    Selection criterion:

        mRMR = relevance - redundancy

    The implementation uses mutual_info_classif for
    relevance and pairwise mutual information for
    redundancy.
    """

    k = min(
        k,
        X.shape[1]
    )

    # --------------------------------------------------------
    # Calculate feature-target relevance
    # --------------------------------------------------------

    relevance = mutual_info_classif(
        X,
        y,
        random_state=random_state
    )

    relevance = pd.Series(
        relevance,
        index=X.columns
    )

    # --------------------------------------------------------
    # Start with the most relevant feature
    # --------------------------------------------------------

    selected = []

    remaining = list(
        X.columns
    )

    first_feature = (
        relevance
        .sort_values(
            ascending=False
        )
        .index[0]
    )

    selected.append(
        first_feature
    )

    remaining.remove(
        first_feature
    )

    # --------------------------------------------------------
    # Greedy mRMR selection
    # --------------------------------------------------------

    while (
        len(selected) < k
        and remaining
    ):

        best_feature = None
        best_score = -np.inf

        for candidate in remaining:

            # ----------------------------------------------
            # Relevance
            # ----------------------------------------------

            candidate_relevance = (
                relevance[
                    candidate
                ]
            )

            # ----------------------------------------------
            # Redundancy
            # ----------------------------------------------

            redundancy_values = []

            for selected_feature in selected:

                pair = X[
                    [
                        candidate,
                        selected_feature
                    ]
                ]

                pair_mi = mutual_info_classif(
                    pair,
                    y=None,
                    discrete_features=False,
                    random_state=random_state
                )

                # The mutual information between two
                # variables is symmetric. We use the
                # first value from the pair calculation
                # only as an approximation.
                #
                # A more direct pairwise MI estimator is
                # difficult with continuous variables.
                redundancy_values.append(
                    pair_mi[0]
                )

            if redundancy_values:

                redundancy = np.mean(
                    redundancy_values
                )

            else:

                redundancy = 0.0

            # ----------------------------------------------
            # mRMR score
            # ----------------------------------------------

            score = (
                candidate_relevance
                -
                redundancy
            )

            if score > best_score:

                best_score = score
                best_feature = candidate

        # --------------------------------------------------
        # Add best feature
        # --------------------------------------------------

        if best_feature is None:
            break

        selected.append(
            best_feature
        )

        remaining.remove(
            best_feature
        )

    return selected


# ============================================================
# ALTERNATIVE / MORE STABLE mRMR IMPLEMENTATION
# ============================================================

def mrmr_selection_discrete(
    X,
    y,
    k=10,
    random_state=42
):
    """
    mRMR implementation for categorical/integer encoded
    clinical features.

    Since the current dataset consists primarily of encoded
    categorical variables, this implementation treats the
    feature columns as discrete variables.

    This is the recommended mRMR function for the current
    maternal-health dataset.
    """

    k = min(
        k,
        X.shape[1]
    )

    feature_names = X.columns.tolist()

    # --------------------------------------------------------
    # Relevance: MI(feature, target)
    # --------------------------------------------------------

    relevance_values = mutual_info_classif(
        X,
        y,
        discrete_features=True,
        random_state=random_state
    )

    relevance = dict(
        zip(
            feature_names,
            relevance_values
        )
    )

    # --------------------------------------------------------
    # Pre-compute pairwise mutual information
    # --------------------------------------------------------

    redundancy = {}

    for i, feature_a in enumerate(
        feature_names
    ):

        for feature_b in feature_names[
            i + 1:
        ]:

            pair = X[
                [
                    feature_a,
                    feature_b
                ]
            ]

            # Estimate MI(feature_a, feature_b)
            mi_values = mutual_info_classif(
                pair[
                    [feature_a]
                ],
                pair[
                    feature_b
                ],
                discrete_features=True,
                random_state=random_state
            )

            mi_value = float(
                mi_values[0]
            )

            redundancy[
                (feature_a, feature_b)
            ] = mi_value

            redundancy[
                (feature_b, feature_a)
            ] = mi_value

    # --------------------------------------------------------
    # First feature: maximum relevance
    # --------------------------------------------------------

    first_feature = max(
        relevance,
        key=relevance.get
    )

    selected = [
        first_feature
    ]

    remaining = [
        feature
        for feature in feature_names
        if feature != first_feature
    ]

    # --------------------------------------------------------
    # Greedy mRMR
    # --------------------------------------------------------

    while (
        len(selected) < k
        and remaining
    ):

        best_feature = None
        best_score = -np.inf

        for candidate in remaining:

            candidate_relevance = (
                relevance[candidate]
            )

            redundancy_values = []

            for selected_feature in selected:

                redundancy_values.append(
                    redundancy.get(
                        (
                            candidate,
                            selected_feature
                        ),
                        0.0
                    )
                )

            if redundancy_values:

                candidate_redundancy = np.mean(
                    redundancy_values
                )

            else:

                candidate_redundancy = 0.0

            mrmr_score = (
                candidate_relevance
                -
                candidate_redundancy
            )

            if mrmr_score > best_score:

                best_score = mrmr_score
                best_feature = candidate

        if best_feature is None:
            break

        selected.append(
            best_feature
        )

        remaining.remove(
            best_feature
        )

    return selected


# ============================================================
# BUILD ABLATION FEATURE SETS
# ============================================================

def build_ablation_feature_sets(
    all_features,
    rfecv_features,
    lasso_features,
    mrmr_features
):
    """
    Build the five feature conditions used in the
    ablation study.

    ALL
    RFECV
    LASSO
    MRMR
    COMBINED

    COMBINED = union of RFECV + LASSO + mRMR.
    """

    all_set = set(
        all_features
    )

    rfecv_set = set(
        rfecv_features
    )

    lasso_set = set(
        lasso_features
    )

    mrmr_set = set(
        mrmr_features
    )

    combined_set = (
        rfecv_set
        |
        lasso_set
        |
        mrmr_set
    )

    feature_sets = {

        "ALL":
            sorted(
                all_set
            ),

        "RFECV":
            sorted(
                rfecv_set
            ),

        "LASSO":
            sorted(
                lasso_set
            ),

        "MRMR":
            sorted(
                mrmr_set
            ),

        "COMBINED":
            sorted(
                combined_set
            )
    }

    return feature_sets


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    features,
    path
):
    """
    Save feature names, one per line.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in features:

            f.write(
                feature + "\n"
            )


# ============================================================
# SAVE ABLATION FEATURE SETS
# ============================================================

def save_ablation_feature_sets(
    feature_sets,
    output_dir
):
    """
    Save all feature sets as individual TXT files.
    """

    output_dir = Path(
        output_dir
    )

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
    lasso_features,
    mrmr_features
):
    """
    Create a feature-level selection report.
    """

    report = pd.DataFrame({
        "feature": all_features
    })

    report["RFECV"] = (
        report["feature"]
        .isin(
            rfecv_features
        )
    )

    report["LASSO"] = (
        report["feature"]
        .isin(
            lasso_features
        )
    )

    report["MRMR"] = (
        report["feature"]
        .isin(
            mrmr_features
        )
    )

    report["frequency"] = (
        report[
            [
                "RFECV",
                "LASSO",
                "MRMR"
            ]
        ].sum(
            axis=1
        )
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