"""
src/models/transformer_models.py

       clinicalbert:
         name: medicalai/ClinicalBERT
"""

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)


def get_transformer_checkpoints(config):
    """
    Returns {model_key: checkpoint_string} for every model entry
    in config["models"] that has a usable `name`. Incomplete
    entries (see module docstring) are skipped with a warning.
    """

    models_config = config.get("models", {})

    checkpoints = {}

    for model_key, model_cfg in models_config.items():

        if not isinstance(model_cfg, dict) or not model_cfg.get("name"):

            print(
                f"WARNING: skipping '{model_key}' — no checkpoint "
                f"'name' set in configs/base.yaml under models.{model_key}"
            )

            continue

        checkpoints[model_key] = model_cfg["name"]

    return checkpoints


def load_tokenizer(checkpoint):

    return AutoTokenizer.from_pretrained(checkpoint)


def load_model(checkpoint, num_labels=2, dropout=None):
    """
    dropout, if given, overrides hidden/attention/classifier
    dropout on the pretrained config (config["transformer"]["dropout"]).
    """

    model_config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=num_labels
    )

    if dropout is not None:

        for attr in (
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "classifier_dropout"
        ):

            if hasattr(model_config, attr):
                setattr(model_config, attr, dropout)

    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=model_config
    )