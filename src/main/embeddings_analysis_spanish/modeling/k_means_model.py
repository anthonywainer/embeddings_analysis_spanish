from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

from embeddings_analysis_spanish.evaluation.compute import get_metrics
from embeddings_analysis_spanish.modeling.base_model import BaseModel
from embeddings_analysis_spanish.utils.mapping import LazyDict


class KMeansModel(BaseModel):

    def __init__(self, path: str):
        super().__init__()
        self.max_iter = 200
        self.jobs = 10,
        self.random_state = 73
        self.path = path

    def fit_predict_kmeans(self, embedding: np.ndarray,
                           name: str, cluster_number: int,
                           y_true: np.ndarray, predicted_embedding: Dict,
                           data_metrics: List) -> Tuple:
        """
        Method to configure k-means model and fit-train
        :param embedding: The embedding values
        :param name: Model tag name
        :param cluster_number: The number cluster
        :param y_true: Original labels
        :param predicted_embedding: Predicted result to save in list
        :param data_metrics: Data Metrics result to save in list
        :return: predicted embedding and data metrics
        """
        self.logger.info("-" * 50)
        self.logger.info("-" * 10, f"Model {name}", "-" * 10)
        kmeans = KMeans(
            n_clusters=cluster_number,
            max_iter=self.max_iter,
            n_jobs=self.jobs,
            random_state=self.random_state
        )
        y_predicted = kmeans.fit_predict(embedding)
        predicted_embedding[name] = (embedding, y_predicted, kmeans)

        data_metrics.append(
            list((name,) + get_metrics(y_true.argmax(1), y_predicted))
        )

        return data_metrics, predicted_embedding

    def run(self, dataset_embeddings: LazyDict, save_results: bool = False) -> Tuple:
        """
        Method to run model with params defined
        """
        data_metrics = []
        predicted_embedding = {}
        for name, items in dataset_embeddings.items():
            self.logger.info("-" * 50)
            self.logger.info("-" * 10, f"Dataset {name}", "-" * 10)

            for embedding_name, embedding in items.embeddings.items():
                self.logger.info("-" * 50)
                self.logger.info("-" * 10, f"Embedding {embedding_name}", "-" * 10)
                data_metrics, predicted_embedding = self.fit_predict_kmeans(
                    embedding,
                    f"{name}-{embedding_name}",
                    items.cluster_number,
                    items.y_true,
                    predicted_embedding,
                    data_metrics
                )
                self.logger.info("-" * 50)

        if save_results:
            self.save_data(f"{self.path}/numpy/predicted_embedding_autoencoder", predicted_embedding)

        return data_metrics, predicted_embedding
