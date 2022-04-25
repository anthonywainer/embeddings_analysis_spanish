from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from embeddings_analysis_spanish.utils.mapping import LazyDict


@dataclass
class EmbeddingsModel:
    dataframe: pd.DataFrame
    x_values: np.ndarray
    y_true: np.ndarray
    cluster_number: int
    embeddings: LazyDict
    labels_encoders: Dict
