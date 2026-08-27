from pathlib import Path

from src.data.clean import (
    load_raw_data,
    clean_dataset
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_INPUT = Path(
    "/kaggle/input/datasets/"
    "jannatsultana/maternal3/"
    "PPD_dataset_v2.csv"
)

LOCAL_INPUT = (
    PROJECT_ROOT /
    "src" /
    "raw" /
    "PPD_dataset_v2.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "src" /
    "processed" /
    "cleaned_dataset.csv"
)


def main():

    print("=" * 60)
    print("MATERNAL MENTAL HEALTH CLEANING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------
    # Select input automatically
    # --------------------------------------------------

    if KAGGLE_INPUT.exists():
        input_path = KAGGLE_INPUT
    else:
        input_path = LOCAL_INPUT

    print("\nInput:")
    print(input_path)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    df = load_raw_data(input_path)

    print("\nOriginal dataset:")
    print(df.shape)

    # --------------------------------------------------
    # Clean
    # --------------------------------------------------

    df_clean = clean_dataset(df)

    print("\nClean dataset:")
    print(df_clean.shape)

    # --------------------------------------------------
    # Target distribution
    # --------------------------------------------------

    print("\nRisk distribution:")
    print(df_clean["Risk"].value_counts())

    print("\nRisk proportions:")
    print(
        df_clean["Risk"]
        .value_counts(normalize=True)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_clean.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()