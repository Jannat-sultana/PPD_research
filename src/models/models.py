from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


def get_models():

    return {

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "AdaBoost":
            AdaBoostClassifier(
                random_state=RANDOM_STATE
            ),

        "LightGBM":
            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1
            ),

        "Logistic Regression":
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=RANDOM_STATE
            ),

        "KNN":
            KNeighborsClassifier()
    }