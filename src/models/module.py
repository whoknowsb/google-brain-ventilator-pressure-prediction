from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
from torchmetrics.regression.mean_squared_error import MeanSquaredError


class LitModule(LightningModule):
    def __init__(self, net):
        super().__init__()
        self.save_hyperparameters()

        self.net = net  # hydra.utils.instantiate(model)

        # loss function
        self.train_mae = MeanAbsoluteError()
        self.valid_mae = MeanAbsoluteError()
        self.criterion = torch.nn.functional.l1_loss

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y, m = batch["inputs"], batch["target"], batch["mask"]
        y_hat = self.forward(x)

        if y_hat.size(0) == 2:  # dual loss
            loss_in = self.criterion(y_hat[0][~m], y[~m])
            loss_out = self.criterion(y_hat[1][m], y[m])
            loss = 0.5 * loss_in + 0.5 * loss_out

            y_hat = y_hat[0] * (1 - m.to(y_hat)) + y_hat[1] * m.to(y_hat)
        else:
            loss = self.criterion(y_hat, y)
        return loss, y_hat.detach(), y.detach(), m

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mask = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/mae",
            self.train_mae(preds[~mask], targets[~mask]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mask = self.step(batch)

        # log val metrics
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "valid/mae",
            self.valid_mae(preds[~mask], targets[~mask]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mask = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        self.preds = torch.cat([o["preds"] for o in outputs])

    def configure_optimizers(self):
        raise NotImplementedError()
