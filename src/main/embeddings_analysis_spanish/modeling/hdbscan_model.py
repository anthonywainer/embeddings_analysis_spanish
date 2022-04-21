from typing import List

import numpy as np
from hdbscan.flat import HDBSCAN_flat

from embeddings_analysis_spanish.evaluation.compute import get_metrics
from embeddings_analysis_spanish.modeling.base_model import BaseModel


class HDBSCANModel(BaseModel):

    def get_result_hdbscan(self, embedding: np.ndarray, name: str, cluster_number: int, y_true: np.ndarray) -> List:
        self.logger.info("-" * 50)
        self.logger.info("-" * 10, f"Model {name}", "-" * 10)
        clustered = HDBSCAN_flat(embedding,
                                 metric='euclidean',
                                 cluster_selection_method='eom',
                                 n_clusters=cluster_number)


        labels = clustered.labels_

        return list((name,) + get_metrics(y_true.argmax(1), labels + 1))

    # [self.get_result_hdbscan(*embedding) for embedding in embeddings]