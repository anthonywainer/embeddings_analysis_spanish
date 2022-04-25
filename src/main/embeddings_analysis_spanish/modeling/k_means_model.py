from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

from embeddings_analysis_spanish.evaluation.compute import get_metrics
from embeddings_analysis_spanish.modeling.base_model import BaseModel
from embeddings_analysis_spanish.utils.mapping import LazyDict


class KMeansModel(BaseModel):

    def __init__(self, path: str = "data/results") -> None:
        super().__init__(path)
        self.max_iter = 200
        self.jobs = 10
        self.random_state = 73
        self.model_name = "hdbscan"

    def fit_predict(self, embedding: np.ndarray,
                    name: str, cluster_number: int,
                    y_true: np.ndarray, predicted_embedding: Dict,
                    data_metrics: List) -> Tuple:
        """
        Method to configure k-means model and fit-train
        :param embedding: The embeddings values
        :param name: Model tag name
        :param cluster_number: The number cluster
        :param y_true: Original labels
        :param predicted_embedding: Predicted result to save in list
        :param data_metrics: Data Metrics result to save in list
        :return: predicted embeddings and data metrics
        """
        self.logger.info("-" * 50)
        self.logger.info("-" * 10 + f"Process {name}" + "-" * 10)
        kmeans = KMeans(
            n_clusters=cluster_number,
            max_iter=self.max_iter,
            n_jobs=self.jobs,
            random_state=self.random_state
        )
        y_predicted = kmeans.fit_predict(embedding)
        predicted_embedding[name] = (embedding, y_predicted, kmeans)

        data_metrics.append(
            list((name,) + get_metrics(y_true.argmax(1), y_predicted).to_tuple)
        )

        return data_metrics, predicted_embedding

    def run(self, dataset_embeddings: LazyDict, save_results: bool = False) -> Tuple:
        return super().run(dataset_embeddings, save_results)
