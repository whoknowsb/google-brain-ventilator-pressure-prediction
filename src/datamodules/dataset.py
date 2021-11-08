import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class VPPDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        df = df.reset_index(drop=True)
        self.target = df["pressure"].to_numpy()
        self.mask = df["mask"].to_numpy()
        self.inputs = df.drop(
            ["pressure", "breath_id", "fold", "id", "mask"], axis=1, errors="ignore"
        ).to_numpy()

    def __len__(self):
        return len(self.target) // 80

    def __getitem__(self, idx):
        target = self.target[slice(idx * 80, (idx + 1) * 80)]
        inputs = self.inputs[slice(idx * 80, (idx + 1) * 80)]
        mask = self.mask[slice(idx * 80, (idx + 1) * 80)]

        return {
            "target": torch.tensor(target, dtype=torch.float32),
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
