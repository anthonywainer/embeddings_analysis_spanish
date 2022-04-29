import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class BBCCleaning(BaseCleaning):
    """
    Cooking bbc_news dataset
    """

    @staticmethod
    def sample_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[:, 'clean_news'] = dataframe.content.apply(processing_words)
        dataframe["id"] = range(0, len(dataframe))

        dataframe = dataframe.drop_duplicates(subset='clean_news', keep="last")
        return dataframe.sort_values(by=["category"])

    def process(self) -> None:
        dataframe = self.read_dataframe(f"{self.path}/translated/bbc_news_es.xlsx")
        dataframe = self.sample_df(dataframe)

        self.write_dataframe(dataframe, f"{self.path}/processed/bbc_news_processed.xlsx")
