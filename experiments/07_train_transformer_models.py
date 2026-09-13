from pathlib import Path

import pandas as pd
import torch
import yaml

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.models.transformer_models import (
    get_transformer_checkpoints,
    load_tokenizer,
    load_model
)
from src.models.transformer_dataset import TextifiedDataset
from src.models.transformer_train import fine_tune
from src.evaluation.transformer_evaluate import (
    predict,
    calculate_metrics
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "configs" / "base.yaml"

TEXT_DIR = ROOT / "src" / "processed" / "text"
TRAIN_TEXT_PATH = TEXT_DIR / "train_text.csv"
TEST_TEXT_PATH = TEXT_DIR / "test_text.csv"

VAL_SIZE = 0.15


def load_config():

    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    target = config["data"]["target"]
    seed = config["seed"]
    text_column = config["text"].get("text_column", "text")
    transformer_cfg = config["transformer"]

    results_dir = ROOT / config["output"]["transformers"]
    checkpoints_dir = results_dir / "checkpoints"

    print("=" * 70)
    print("TRANSFORMER ABLATION")
    print("=" * 70)

    if not TRAIN_TEXT_PATH.exists() or not TEST_TEXT_PATH.exists():

        raise FileNotFoundError(
            "Textified data not found. Run:\n"
            "python -m experiments.06_textify_dataset"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    train_full = pd.read_csv(TRAIN_TEXT_PATH)
    test_df = pd.read_csv(TEST_TEXT_PATH)

    print(f"Train rows: {len(train_full)}")
    print(f"Test rows:  {len(test_df)}")

    # --------------------------------------------------
    # Carve a validation split OUT OF TRAINING ONLY.
    # The test set stays untouched for final reporting,
    # matching the fixed-split contract from step 1.
    # --------------------------------------------------

    train_df, val_df = train_test_split(
        train_full,
        test_size=VAL_SIZE,
        stratify=train_full[target],
        random_state=seed
    )

    checkpoints = get_transformer_checkpoints(config)

    if not checkpoints:
        raise ValueError(
            "No usable model checkpoints found in configs/base.yaml "
            "under `models:`. See src/models/transformer_models.py "
            "for the known config issues to fix first."
        )

    print(f"\nModels to run: {list(checkpoints)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_key, checkpoint in checkpoints.items():

        print("\n" + "=" * 70)
        print(f"MODEL: {model_key} ({checkpoint})")
        print("=" * 70)

        tokenizer = load_tokenizer(checkpoint)

        model = load_model(
            checkpoint,
            num_labels=2,
            dropout=transformer_cfg.get("dropout")
        )

        train_dataset = TextifiedDataset(
            texts=train_df[text_column],
            labels=train_df[target],
            tokenizer=tokenizer,
            max_length=transformer_cfg["max_length"]
        )

        val_dataset = TextifiedDataset(
            texts=val_df[text_column],
            labels=val_df[target],
            tokenizer=tokenizer,
            max_length=transformer_cfg["max_length"]
        )

        test_dataset = TextifiedDataset(
            texts=test_df[text_column],
            labels=test_df[target],
            tokenizer=tokenizer,
            max_length=transformer_cfg["max_length"]
        )

        # ------------------------------------------------
        # Fine-tune
        # ------------------------------------------------

        trainer = fine_tune(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=checkpoints_dir / model_key,
            transformer_config=transformer_cfg,
            seed=seed
        )

        # ------------------------------------------------
        # Evaluate on the held-out FIXED test set
        # ------------------------------------------------

        test_loader = DataLoader(
            test_dataset,
            batch_size=transformer_cfg["eval_batch_size"]
        )

        y_true, y_pred, y_prob = predict(
            trainer.model,
            test_loader,
            device
        )

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        print("\nTest metrics:")

        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        all_results.append({
            "model": model_key,
            "checkpoint": checkpoint,
            **metrics
        })

        predictions = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probability": y_prob
        })

        predictions.to_csv(
            results_dir / f"{model_key}_predictions.csv",
            index=False
        )

        # Free GPU memory before the next checkpoint
        del model, trainer
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Save combined results table
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    results_path = results_dir / "transformer_ablation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("TRANSFORMER ABLATION SUMMARY")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to:\n{results_path}")


if __name__ == "__main__":
    main()