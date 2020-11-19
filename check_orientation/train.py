import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict as Adict
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from check_orientation.dataloaders import ClassificationDataset

train_image_path = Path(os.environ["TRAIN_IMAGE_PATH"])
val_image_path = Path(os.environ["VAL_IMAGE_PATH"])


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class CheckOrientation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = object_from_dict(self.hparams.model)
        self.loss = object_from_dict(self.hparams.loss)
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, batch):
        return self.model(batch)

    def setup(self, stage=0):  # pylint: disable=W0613
        self.train_image_paths = sorted(train_image_path.glob("*.jpg"))
        self.val_image_paths = sorted(val_image_path.glob("*.jpg"))

        print("Len train samples = ", len(self.train_image_paths))
        print("Len val samples = ", len(self.val_image_paths))

    def train_dataloader(self):
        train_aug = from_dict(self.hparams.train_aug)

        if "epoch_length" not in self.hparams.train_parameters:
            epoch_length = None
        else:
            epoch_length = self.hparams.train_parameters.epoch_length

        result = DataLoader(
            ClassificationDataset(self.train_image_paths, train_aug, epoch_length),
            batch_size=self.hparams.train_parameters.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams.val_aug)

        result = DataLoader(
            ClassificationDataset(self.val_image_paths, val_aug, length=None),
            batch_size=self.hparams.val_parameters.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))
        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparams.scheduler, optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        features = batch["features"]
        targets = batch["targets"]

        logits = self.forward(features)

        loss = self.loss(logits, targets)

        self.log("lr", self._get_current_lr(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log(
            "train_acc", self.train_accuracy(logits, targets), on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id):  # pylint: disable=W0613
        features = batch["features"]
        targets = batch["targets"].long()

        logits = self.forward(features)
        loss = self.loss(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "val_acc", self.val_accuracy(logits, targets), on_step=False, on_epoch=True, prog_bar=False, logger=True
        )


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    pl.trainer.seed_everything(hparams.seed)

    pipeline = CheckOrientation(hparams)

    Path(hparams.checkpoint_callback.filepath).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams.trainer,
        logger=WandbLogger(hparams.experiment_name),
        checkpoint_callback=object_from_dict(hparams.checkpoint_callback),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
