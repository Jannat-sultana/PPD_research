from pathlib import Path
import pandas as pd

from src.data.split import create_fixed_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT /
    "src" /
    "processed" /
    "cleaned_dataset.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT /
    "src" /
    "processed" /
    "splits"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Risk"


def main():

    print("=" * 60)
    print("CREATE FIXED TRAIN / TEST SPLIT")
    print("=" * 60)

    # --------------------------------------------------
    # Load cleaned dataset
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print("\nCleaned dataset:")
    print(df.shape)

    # --------------------------------------------------
    # Create split
    # --------------------------------------------------

    train_df, test_df = create_fixed_split(
        df,
        target=TARGET,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTraining:")
    print(train_df.shape)

    print("\nTesting:")
    print(test_df.shape)

    # --------------------------------------------------
    # Target distributions
    # --------------------------------------------------

    print("\nTraining Risk distribution:")
    print(train_df[TARGET].value_counts())

    print("\nTesting Risk distribution:")
    print(test_df[TARGET].value_counts())

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nSaved:")
    print(TRAIN_PATH)
    print(TEST_PATH)

    print("\nFixed split created successfully.")

    print("\nIMPORTANT:")
    print(
        "Do NOT create another train/test split "
        "in downstream experiments."
    )


if __name__ == "__main__":
    main()