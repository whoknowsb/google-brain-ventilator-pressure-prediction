from itertools import product

import numpy as np
import pandas as pd

def RC_dummies(df):
    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df = pd.get_dummies(df)
    print(df.head())
    return df



