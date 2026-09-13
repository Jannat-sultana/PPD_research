"""
src/models/transformer_train.py

Fine-tunes one BERT-family checkpoint on the textified
maternal-health narratives. All hyperparameters come from
configs/base.yaml's `transformer:` section.
"""

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

import numpy as np

from sklearn.metrics import f1_score


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    return {
        "f1": f1_score(labels, preds, zero_division=0)
    }


def fine_tune(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    transformer_config,
    seed=42
):
    """
    transformer_config is config["transformer"] from base.yaml:
    batch_size, eval_batch_size, learning_rate, weight_decay,
    epochs, warmup_ratio, early_stopping_patience.

    (max_length and dropout are consumed earlier, by the dataset
    and load_model respectively — not needed here.)

    NOTE: eval_dataset must be a validation slice carved out of
    TRAINING data only (see experiments/07_train_transformer_models.py)
    — never the held-out test set, to match the fixed-split
    contract from step 1.
    """

    # NOTE: `eval_strategy` is the current transformers kwarg name;
    # if you're on transformers < 4.41 rename it to `evaluation_strategy`.
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=transformer_config["epochs"],
        per_device_train_batch_size=transformer_config["batch_size"],
        per_device_eval_batch_size=transformer_config["eval_batch_size"],
        learning_rate=transformer_config["learning_rate"],
        weight_decay=transformer_config["weight_decay"],
        warmup_ratio=transformer_config["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        seed=seed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    transformer_config["early_stopping_patience"]
                )
            )
        ]
    )

    trainer.train()

    return trainer