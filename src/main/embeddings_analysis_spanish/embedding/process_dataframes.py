from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from embeddings_analysis_spanish.embedding.base_embedding import BaseEmbedding
from embeddings_analysis_spanish.utils.logger import Logger
from embeddings_analysis_spanish.utils.mapping import LazyDict


@dataclass
class DataFrameEmbeddings:
    dataframe: pd.DataFrame
    x_values: np.ndarray
    y_true: np.ndarray
    cluster_number: int
    embeddings: LazyDict
    labels_encoders: Dict


class ProcessDataFrames(Logger):
    """
    Space to read dataset processed and prepare the inputs to train models
    """

    def __init__(self, path: str = "data"):
        super().__init__()
        self.label_encoder = LabelEncoder()
        self.path = path
        self.dataset_path = f"{path}/dataset/processed"
        self.base_embedding = BaseEmbedding(f"{path}/gensim", f"{path}/numpy")

    def get_dataframe(self, name: str, data: Tuple) -> DataFrameEmbeddings:
        self.logger.info(f"Getting Dataset from {self.dataset_path}/{name}_processed.xlsx")
        dataset = pd.read_excel(f"{self.dataset_path}/{name}_processed.xlsx")

        x_values = dataset[data[0]].astype(str)
        y_true = to_categorical(self.label_encoder.fit_transform(dataset[data[1]]))

        cluster_number = len(self.label_encoder.classes_)
        labels_encoders = dict(enumerate(self.label_encoder.classes_))
        self.logger.debug(f"Features: {labels_encoders}")

        embeddings = LazyDict({
            embedding: (
                self.base_embedding.extract_embedding,
                embedding,
                name,
                x_values[0:2]
            ) for embedding in self.base_embedding.embeddings
        })

        return DataFrameEmbeddings(dataset, x_values, y_true, cluster_number, embeddings, labels_encoders)

    @property
    def dataframes(self) -> Dict:
        return {
            "bbc_news": ("clean_content", "category"),
            "complaints": ("clean_complaint", "product"),
            "food_reviews": ("clean_review", "sentiment"),
            "imdb_reviews": ("clean_review", "sentiment"),
            "paper_scopus": ("clean_abstract", "category")
        }

    def start_process(self) -> LazyDict:
        return LazyDict({
            dataframe: (self.get_dataframe, (dataframe, data)) for dataframe, data in self.dataframes.items()
        })
