import numpy as np
import torch

from tqdm.auto import tqdm


@torch.no_grad()
def predict(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in tqdm(
        loader,
        desc="Evaluating"
    ):

        labels = (
            batch["labels"]
            .to(device)
        )

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(
            **inputs
        )

        logits = outputs["logits"]

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )