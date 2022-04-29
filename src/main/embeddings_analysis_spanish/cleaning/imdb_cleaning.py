from typing import List

import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class IMDBCleaning(BaseCleaning):

    def pre_cleaning(self) -> None:
        import cld3
        dataset = pd.read_csv(f"{self.path}/original/IMDB Dataset SPANISH.csv")
        dataset.loc[:, 'clean_review'] = dataset.review_es.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='clean_review', keep="last")
        dataset.sort_index(inplace=True)

        dataset = dataset[dataset.review_es.apply(
            lambda x: cld3.get_language(x).language != "en"
        )][["review_es", "sentimiento", "clean_review_es"]]

        dataset = dataset.rename(columns={"clean_review_es": "clean_review", "sentimiento": "sentiment"})
        self.write_dataframe(dataset, f"{self.path}/translated/IMDBPreClean.xlsx")

    @property
    def __columns(self) -> List:
        return ["id", "review", "clean_review", "sentiment", "words_len"]

    def sample_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["words_len"] = dataframe.clean_review.apply(lambda c: len(set(c.split())))
        dataframe["id"] = range(0, len(dataframe))

        positive_df = dataframe[(dataframe.words_len >= 200) & (dataframe.sentiment == "positivo")][0:1500]
        negative_df = dataframe[(dataframe.words_len >= 200) & (dataframe.sentiment == "negativo")][0:1500]

        dataframe = pd.concat([positive_df, negative_df])[self.__columns]

        return dataframe.sort_values(by=['sentiment'])

    def process(self) -> None:
        dataframe = pd.read_excel(f"{self.path}/translated/imdb_reviews_es.xlsx")
        dataframe = self.sample_df(dataframe)
        self.write_dataframe(dataframe, f"{self.path}/processed/imdb_reviews_processed.xlsx")
