import torch

from tqdm.auto import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device
):

    model.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training"
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(
            **batch
        )

        loss = outputs["loss"]

        loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss /
        len(loader)
    )