import torch

from torch import nn

from transformers import (
    AutoModel,
    AutoConfig
)


class TransformerClassifier(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_labels=2,
        dropout=0.2
    ):

        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name
        )

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
            if token_type_ids is not None
            else None,
            output_attentions=True
        )

        # CLS representation
        pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(
            self.dropout(pooled)
        )

        loss = None

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.last_hidden_state,
            "attentions": outputs.attentions
        }