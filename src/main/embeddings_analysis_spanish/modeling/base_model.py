from typing import Union, Tuple

import numpy as np
import pandas as pd

from embeddings_analysis_spanish.abstracts.abstract_model import AbstractModel
from embeddings_analysis_spanish.utils.logger import Logger
from embeddings_analysis_spanish.utils.mapping import LazyDict


class BaseModel(AbstractModel, Logger):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.model_name = None

    def _show_result(self, data) -> pd.DataFrame:
        df = pd.DataFrame(
            data,
            columns=["Process", "ACC", "NMI", "ARI"]
        ).sort_values(by=["ACC"], ascending=False)

        self.logger.info(df.to_string())

        return df

    @staticmethod
    def __save_data(path, predicted_embedding: Union) -> None:
        np.savez(
            path,
            np.array(
                tuple(
                    {k: v[:2] for k, v in predicted_embedding.items()}.items()
                )
            )
        )

    def fit_predict(self, *args) -> Tuple:
        pass

    def run(self, dataset_embeddings: LazyDict, save_results: bool = False) -> Tuple:
        """
        Method to run model with params defined
        """
        data_metrics = []
        predicted_embedding = {}
        for name, items in dataset_embeddings.items():
            self.logger.info("-" * 50)
            self.logger.info("-" * 10 + f"Dataset {name}" + "-" * 10)

            for embedding_name, embedding in items.embeddings.items():
                self.logger.info("-" * 50)
                self.logger.info("-" * 10 + f"Embedding {embedding_name}" + "-" * 10)
                data_metrics, predicted_embedding = self.fit_predict(
                    embedding,
                    f"{name}-{embedding_name}",
                    items.cluster_number,
                    items.y_true,
                    predicted_embedding,
                    data_metrics
                )
                self.logger.info("-" * 50)

        if save_results:
            self.__save_data(f"{self.path}/numpy/predicted_embedding_{self.model_name}", predicted_embedding)

        return data_metrics, predicted_embedding
