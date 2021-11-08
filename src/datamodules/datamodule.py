import importlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


def create_stratified_folds(train, splitter):
    breath = train.groupby("breath_id").median().reset_index()
    n_breath = len(breath.pressure)
    n_quant = np.log2(n_breath).astype(int)

    quantiles = pd.qcut(breath.pressure, n_quant)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(quantiles)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(breath, y)):
        breath.loc[valid_idx, "fold"] = fold
    breath.fold = breath.fold.astype(int)
    breath = breath[["breath_id", "fold"]]
    train = train.merge(breath)
    return train


def create_folds(train, splitter):
    breath = train.groupby("breath_id").first().reset_index()[["breath_id"]]
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(breath, breath)):
        breath.loc[valid_idx, "fold"] = fold
    breath.fold = breath.fold.astype(int)
    breath = breath[["breath_id", "fold"]]
    train = train.merge(breath)
    return train


class VPPDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/input/ventilator-pressure-prediction",
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        submission_file: str = "sample_submission.csv",
        fold: int = 0,
        batch_size: int = 0,
        dataset=None,
        splitter=None,
        featurizer=None,
        neighborizer=None,
        normalizer=None,
        shifter=False,
        save_df=False,
        pseudo=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / train_file
        self.test_path = self.data_dir / test_file
        self.submission_path = self.data_dir / submission_file

        self.fold = fold
        self.batch_size = batch_size
        self.dataset = dataset
        splitter = hydra.utils.instantiate(splitter)
        normalizer = hydra.utils.instantiate(normalizer)

        self.oof_df = None
        self.pred_df = None

        train = pd.read_csv(self.train_path, index_col="id").reset_index()
        test = pd.read_csv(self.test_path, index_col="id").reset_index()
        train["mask"] = train.u_out
        test["mask"] = test.u_out

        train = create_folds(train, splitter)

        if pseudo:
            print("Loading pseudo")
            pseudo = pd.read_csv(pseudo, index_col="id").reset_index()
            pseudo["mask"] = pseudo.u_out
            pseudo["fold"] = -1
            train = train.append(pseudo, ignore_index=True)
            print("After pseudo:")
            print(train)

        if shifter:
            print("Creating shifts")
            self.train_shifts = get_shifts(train)
            self.test_shifts = get_shifts(test)
            train = apply_shifts(train, self.train_shifts)
            test = apply_shifts(test, self.test_shifts)

        if neighborizer:
            print("Creating neighbors:", "train:", train.shape, "test:", test.shape)
            train["time_id"] = train.groupby("breath_id").cumcount()
            test["time_id"] = test.groupby("breath_id").cumcount()
            nn = pd.concat([train, test]).reset_index(drop=True)
            nn = hydra.utils.instantiate(neighborizer, nn)
            train = train.merge(nn, on=["breath_id", "time_id"])
            test = test.merge(nn, on=["breath_id", "time_id"])
            train = train.drop("time_id", axis=1)
            test = test.drop("time_id", axis=1)
            print("After neighbors:", "train:", train.shape, "test:", test.shape)
            print(train.head())
            print(test.head())
            del nn

        if featurizer:
            print("Creating features")
            train = hydra.utils.instantiate(featurizer, train)
            test = hydra.utils.instantiate(featurizer, test)

        if normalizer:
            print(f"Nomalizing using {normalizer}")
            cols = [
                c
                for c in train.columns
                if c not in ["pressure", "breath_id", "fold", "id", "mask"]
            ]
            normalizer.fit(train[cols])
            train[cols] = normalizer.transform(train[cols])
            test[cols] = normalizer.transform(test[cols])

        print("Final train shape:", train.shape)
        print("Final train shape:", train.shape)
        train_df = train[train.fold != self.fold]
        valid_df = train[train.fold == self.fold]
        train_ds = hydra.utils.instantiate(self.dataset, train_df)
        valid_ds = hydra.utils.instantiate(self.dataset, valid_df)
        test_ds = hydra.utils.instantiate(self.dataset, test.assign(pressure=0.0))

        self.oof_df = valid_df[["id"]].reset_index(drop=True)
        self.pred_df = test[["id"]].reset_index(drop=True)

        if save_df:
            self.train = train
            self.test = test
        else:
            del train
            del test

        self.train_dl = DataLoader(
            dataset=train_ds,
            batch_size=self.batch_size,
            num_workers=32,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

        self.valid_dl = DataLoader(
            dataset=valid_ds,
            batch_size=self.batch_size,
            num_workers=32,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        self.test_dl = DataLoader(
            dataset=test_ds,
            batch_size=self.batch_size,
            num_workers=32,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def test_dataloader(self):
        return self.test_dl
