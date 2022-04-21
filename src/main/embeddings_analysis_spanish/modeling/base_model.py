from typing import Union

import numpy as np
import pandas as pd

from embeddings_analysis_spanish.utils.logger import Logger


class BaseModel(Logger):
    @staticmethod
    def show_result(data):
        pd.DataFrame(
            data,
            columns=["Embedding", "ACC", "NMI", "ARI"]
        ).sort_values(by=["ACC"], ascending=False)

    @staticmethod
    def save_data(path, predicted_embedding: Union):
        np.savez(
            path,
            np.array(
                tuple(
                    {k: v[:2] for k, v in predicted_embedding.items()}.items()
                )
            )
        )
