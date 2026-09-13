from pathlib import Path

import pandas as pd
import yaml

from src.data.textify import save_text_dataset


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs" / "base.yaml"

# Not currently in base.yaml — add text.train_output_path /
# text.test_output_path there if you'd rather keep this
# config-driven end to end.
TEXT_OUTPUT_DIR = ROOT / "src" / "processed" / "text"


def load_config():

    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    target = config["data"]["target"]
    text_cfg = config["text"]

    train_path = ROOT / config["data"]["train_path"]
    test_path = ROOT / config["data"]["test_path"]

    print("=" * 70)
    print("TEXTIFICATION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------
    # Load the SAME fixed split used everywhere else
    # --------------------------------------------------

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"\nTrain rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")

    if text_cfg.get("include_target"):
        print(
            "\nWARNING: text.include_target is TRUE. This adds a "
            f"'{text_cfg['text_column']}_with_target' column that "
            "states the Risk label in plain text. Never use that "
            "column as model input — only the plain "
            f"'{text_cfg['text_column']}' column."
        )

    # --------------------------------------------------
    # Textify (uses ALL cleaned columns by default; to
    # textify only the COMBINED/best feature set from step 1
    # instead, pass feature_columns= to save_text_dataset)
    # --------------------------------------------------

    train_text = save_text_dataset(
        train_df,
        TEXT_OUTPUT_DIR / "train_text.csv",
        include_target=text_cfg.get("include_target", False),
        text_column=text_cfg.get("text_column", "text"),
        separator=text_cfg.get("separator", " ")
    )

    test_text = save_text_dataset(
        test_df,
        TEXT_OUTPUT_DIR / "test_text.csv",
        include_target=text_cfg.get("include_target", False),
        text_column=text_cfg.get("text_column", "text"),
        separator=text_cfg.get("separator", " ")
    )

    print("\nSample narrative (train row 0):")
    print(train_text.iloc[0][text_cfg.get("text_column", "text")])

    print("\nTextification complete.")


if __name__ == "__main__":
    main()