import numpy as np
import pandas as pd
from panns_inference import config


def filtered_argsort(x, thresh):
    (idx,) = np.where(x > thresh)
    return idx[np.argsort(x[idx])]


def to_df(framewise_output) -> pd.DataFrame:
    """Visualization of sound event detection result.
    Args:
      framewise_output: (time_steps, classes_num)
    """
    classwise_output = np.max(framewise_output, axis=0)
    idxes = filtered_argsort(classwise_output, 0.05)[::-1]
    idxes = idxes[:5]

    if len(idxes) == 0:
        return None

    df = pd.DataFrame()

    df["time"] = np.arange(len(framewise_output[:, 1])) / 100

    for idx in idxes:
        df[config.ix_to_lb[idx]] = framewise_output[:, idx]

    return df.iloc[::32, :]
