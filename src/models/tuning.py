import numpy as np

from sklearn.model_selection import RandomizedSearchCV


RANDOM_STATE = 42


def get_param_distributions():

    return {

        "Decision Tree": {
            "max_depth": [
                None, 5, 10, 20, 30
            ],
            "min_samples_split": [
                2, 5, 10
            ],
            "min_samples_leaf": [
                1, 2, 4
            ]
        },

        "Random Forest": {
            "n_estimators": [
                100, 200, 300
            ],
            "max_depth": [
                None, 10, 20, 30
            ],
            "min_samples_split": [
                2, 5, 10
            ],
            "min_samples_leaf": [
                1, 2, 4
            ]
        },

        "AdaBoost": {
            "n_estimators": [
                50, 100, 150
            ],
            "learning_rate": [
                0.01, 0.1, 1.0
            ]
        },

        "LightGBM": {
            "num_leaves": [
                15, 31, 62, 127
            ],
            "learning_rate": [
                0.01, 0.05, 0.1
            ],
            "n_estimators": [
                100, 200, 300
            ]
        },

        "XGBoost": {
            "n_estimators": [
                100, 200, 300
            ],
            "learning_rate": [
                0.01, 0.05, 0.1
            ],
            "max_depth": [
                3, 6, 9
            ]
        },

        "Logistic Regression": {
            "C": np.logspace(-4, 4, 20),
            "solver": [
                "liblinear",
                "lbfgs",
                "saga"
            ],
            "max_iter": [
                500,
                1000,
                2000
            ]
        },

        "SVM": {
            "C": [
                0.1,
                1,
                10,
                100
            ],
            "kernel": [
                "linear",
                "rbf"
            ],
            "gamma": [
                "scale",
                "auto"
            ]
        },

        "KNN": {
            "n_neighbors": [
                3, 5, 7, 9, 11
            ],
            "weights": [
                "uniform",
                "distance"
            ],
            "metric": [
                "euclidean",
                "manhattan"
            ]
        }
    }


def tune_model(
    model,
    param_distribution,
    X_train,
    y_train
):

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distribution,
        n_iter=10,
        scoring="f1",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    return search