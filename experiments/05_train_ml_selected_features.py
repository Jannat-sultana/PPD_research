from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(
    __file__
).resolve().parents[1]


def load_config():

    with open(
        ROOT / "configs" / "base.yaml"
    ) as f:

        return yaml.safe_load(f)


def main():

    config = load_config()

    target = config["data"]["target"]

    train = pd.read_csv(
        ROOT /
        config["data"]["train_path"]
    )

    test = pd.read_csv(
        ROOT /
        config["data"]["test_path"]
    )

    feature_file = (
        ROOT /
        config["data"]
        ["selected_features_path"]
    )

    with open(feature_file) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        "Selected features:"
    )

    print(features)

    train_selected = train[
        features + [target]
    ].copy()

    test_selected = test[
        features + [target]
    ].copy()

    train_path = (
        ROOT /
        config["data"]
        ["selected_train_path"]
    )

    test_path = (
        ROOT /
        config["data"]
        ["selected_test_path"]
    )

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_selected.to_csv(
        train_path,
        index=False
    )

    test_selected.to_csv(
        test_path,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(train_path)

    print(test_path)


if __name__ == "__main__":

    main()