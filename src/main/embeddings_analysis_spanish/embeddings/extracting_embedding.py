from typing import List

import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from embeddings_analysis_spanish.embeddings.base_embedding import BaseEmbedding
from embeddings_analysis_spanish.models.dataframe_model import DataframeModel
from embeddings_analysis_spanish.models.embeddings_model import EmbeddingsModel
from embeddings_analysis_spanish.utils.mapping import LazyDict


class ExtractingEmbedding(BaseEmbedding):
    """
    Base Embedding
    """

    def __init__(self, gensim_path: str = "data/gensim", numpy_path: str = "data/numpy") -> None:
        """
        Init embeddings extraction
        :param gensim_path: Path where is vectors Gensim
        :param numpy_path: Path to save or load vector numpy
        """

        super().__init__()
        self.gensim_path = gensim_path
        self.numpy_path = numpy_path
        self.label_encoder = LabelEncoder()

    def get_embedding(self, dataframe: pd.DataFrame, data: DataframeModel, label_encoder) -> EmbeddingsModel:
        x_values = dataframe[data.clean_field].astype(str)
        y_true = to_categorical(label_encoder.fit_transform(dataframe[data.category_field]))

        cluster_number = len(label_encoder.classes_)
        labels_encoders = dict(enumerate(label_encoder.classes_))
        self.logger.debug(f"Features: {labels_encoders}")

        embeddings = LazyDict({
            embedding: (
                self.extract_embedding,
                (embedding, data.dataframe_name, x_values)
            ) for embedding in self.embeddings_analysis
        })

        return EmbeddingsModel(dataframe, x_values, y_true, cluster_number, embeddings, labels_encoders)

    def start_process(self, datasets: LazyDict, dataframes_params: List[DataframeModel]) -> LazyDict:
        return LazyDict({
            data.dataframe_name: (
                self.get_embedding, (datasets[data.dataframe_name], data, self.label_encoder)
            ) for data in dataframes_params
        })
