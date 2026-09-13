from pathlib import Path

import pandas as pd

from sklearn.feature_selection import (
    RFECV,
    RFE,
    SelectKBest,
    f_classif
)

from sklearn.ensemble import RandomForestClassifier


def get_feature_columns(df, target):

    return [
        column
        for column in df.columns
        if column != target
    ]


def rfecv_selection(
    X,
    y,
    cv=5,
    step=1,
    min_features_to_select=5
):

    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
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


def rfe_selection(
    X,
    y,
    n_features
):

    estimator = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
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


def anova_selection(
    X,
    y,
    k
):

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


def save_features(
    features,
    path
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w") as f:

        for feature in features:

            f.write(
                feature + "\n"
            )