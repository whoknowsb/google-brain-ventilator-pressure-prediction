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


def train(config: DictConfig) -> Optional[float]:

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    log.info(model)

    # Init lightning optim
    log.info(
        f"Instantiating optimizer <{config.optim.optimizer._target_}> and scheduler <{config.optim.scheduler.scheduler._target_}>"
    )
    optimizer = hydra.utils.instantiate(config.optim.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(config.optim.scheduler, scheduler={"optimizer": optimizer})
    scheduler = OmegaConf.to_container(scheduler, resolve=True)  # convert to dict

    def configure_optimizers():
        return [optimizer], [scheduler]

    model.configure_optimizers = configure_optimizers

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):

        log.info("Starting oof!")
        trainer.test(test_dataloaders=trainer.datamodule.val_dataloader())
        oof_df = trainer.datamodule.oof_df.copy()
        oof_df["pressure"] = trainer.model.preds.cpu().numpy().astype(float).flatten()
        oof_df.to_csv(f"{config.work_dir}/{config.log_dir}/oof_df.csv", index=False)
        log.info(f"oof shape: {oof_df.shape}")
        log.info(f"oof head: {oof_df.head()}")

        log.info("Starting testing!")
        trainer.test(test_dataloaders=trainer.datamodule.test_dataloader())
        pred_df = trainer.datamodule.pred_df.copy()
        pred_df["pressure"] = trainer.model.preds.cpu().numpy().astype(float).flatten()
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
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
