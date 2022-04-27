from typing import List, Dict, Tuple

import numpy as np
from hdbscan.flat import HDBSCAN_flat

from embeddings_analysis_spanish.evaluation.compute import get_metrics
from embeddings_analysis_spanish.modeling.base_model import BaseModel
from embeddings_analysis_spanish.utils.mapping import LazyDict


class HDBSCANModel(BaseModel):
    def __init__(self, path: str = "data/results") -> None:
        super().__init__(path)
        self.model_name = "hdbscan"

    def fit_predict(self, embedding: np.ndarray,
                    name: str, cluster_number: int,
                    y_true: np.ndarray, predicted_embedding: Dict,
                    data_metrics: List) -> Tuple:
        """
        Method to configure HDBSCAN model and fit-train
        :param embedding: The embeddings values
        :param name: Model tag name
        :param cluster_number: The number cluster
        :param y_true: Original labels
        :param predicted_embedding: Predicted result to save in list
        :param data_metrics: Data Metrics result to save in list
        :return: predicted embeddings and data metrics
        """
        self.logger.info("-" * 50)
        self.logger.info("-" * 10 + f"Model {name}" + "-" * 10)
        model = HDBSCAN_flat(
            embedding,
            metric='euclidean',
            cluster_selection_method='eom',
            n_clusters=cluster_number,
            prediction_data=True
        )
        y_predicted = model.labels_ + 1
        predicted_embedding[name] = (embedding, y_predicted, model)

        data_metrics.append(
            list((name,) + get_metrics(y_true.argmax(1), y_predicted).to_tuple)
        )

        return data_metrics, predicted_embedding

    def run(self, dataset_embeddings: LazyDict, save_results: bool = False) -> Tuple:
        return super().run(dataset_embeddings, save_results)
