from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from embeddings_analysis_spanish.models.dataframe_model import DataframeModel
from embeddings_analysis_spanish.utils.logger import Logger
from embeddings_analysis_spanish.utils.mapping import LazyDict


class ProcessDataFrames(Logger):
    """
    Space to read dataset processed and prepare the inputs to train models
    """

    def __init__(self, path: str = "data") -> None:
        super().__init__()
        self.label_encoder = LabelEncoder()
        self.path = path
        self.dataset_path = f"{path}/dataset/processed"

    def read_dataframe(self, name: str) -> pd.DataFrame:
        self.logger.info(f"Reading Dataset from {self.dataset_path}/{name}_processed.xlsx")
        return pd.read_excel(f"{self.dataset_path}/{name}_processed.xlsx")

    @property
    def dataframes_params(self) -> List[DataframeModel]:
        return [
            DataframeModel("bbc_news", "clean_content", "category"),
            DataframeModel("complaints", "clean_complaint", "product"),
            DataframeModel("food_reviews", "clean_review", "sentiment"),
            DataframeModel("imdb_reviews", "clean_review", "sentiment"),
            DataframeModel("paper_scopus", "clean_abstract", "category")
        ]

    @property
    def dataframes(self) -> LazyDict:
        return LazyDict({
            dataframe.dataframe_name: (
                self.read_dataframe, (dataframe.dataframe_name,)
            ) for dataframe in self.dataframes_params
        })
