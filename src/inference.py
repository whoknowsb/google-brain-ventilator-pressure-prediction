from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)

OmegaConf.register_new_resolver("eval", lambda x: eval(str(x)))


def inference(config: DictConfig) -> Optional[float]:

    # Loading best model
    log.info("Looking for best model")

    ckpts = Path(config["work_dir"]) / config["log_dir"]
    ckpts = list(ckpts.glob("**/*.ckpt"))
    ckpts = {ckpt: float(ckpt.stem[-6:]) for ckpt in ckpts}

    log.info("\033[95mALL CHECKPOINTS MODEL")
    log.info(f"{ckpts}")

    best_ckpt = min(ckpts, key=ckpts.get)
    best_ckpt = sorted(ckpts, key=ckpts.get, reverse=False)[0]

    log.info("\033[95mBEST MODEL")
    log.info(f"\033[95m{best_ckpt.name}")

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    log.info(model)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=[], logger=[], _convert_="partial")

    log.info("Loading best ckpt")
    log.info(f"{best_ckpt}")
    model = model.load_from_checkpoint(checkpoint_path=best_ckpt)

    # Train the model
    log.info("Starting inference!")

    log.info("Starting oof!")
    trainer.test(model, test_dataloaders=datamodule.val_dataloader())
    oof_df = datamodule.oof_df.copy()
    oof_df["pressure"] = model.preds.cpu().numpy().astype(float).flatten()
    oof_df.to_csv(f"{config.work_dir}/{config.log_dir}/oof_df.csv", index=False)
    log.info(f"oof shape: {oof_df.shape}")
    log.info(f"oof head: {oof_df.head()}")

    log.info("Starting testing!")
    trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    pred_df = datamodule.pred_df.copy()
    pred_df["pressure"] = model.preds.cpu().numpy().astype(float).flatten()
    pred_df.to_csv(f"{config.work_dir}/{config.log_dir}/pred_df.csv", index=False)
    log.info(f"preds shape: {pred_df.shape}")
    log.info(f"preds head: {pred_df.head()}")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        logger=[],
    )
