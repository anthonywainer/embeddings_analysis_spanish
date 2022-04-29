from typing import List

import pandas as pd
from textblob import TextBlob

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class FoodCleaning(BaseCleaning):

    def __get_polarity(self, text: str) -> str:
        try:
            t = TextBlob(text)
            return t.translate(from_lang="es", to="en").correct().sentiment.polarity
        except Exception as e:
            self.logger.error("ERROR", e)
            return ""

    def sample_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[:, 'clean_review'] = dataframe.review.apply(processing_words)
        dataframe = dataframe.drop_duplicates(subset='review', keep="last")
        dataframe.sort_index(inplace=True)
        negative = dataframe[dataframe.score.astype("int") < 3][:57000]
        negative["sentiment"] = "negativo"
        neutro = dataframe[dataframe.score.astype("int") == 3][:57000]
        neutro["sentiment"] = "neutro"
        positive = dataframe[dataframe.score.astype("int") > 3][:57000]
        positive["sentiment"] = "positivo"
        dataframe = pd.concat([negative, neutro, positive])[["id_review", "review", "sentiment", "clean_review"]]

        dataframe["words_len"] = dataframe.clean_review.apply(lambda c: self._count_words(c))
        nd = dataframe[dataframe.words_len >= 60]
        nd['polarity'] = nd['review'].apply(lambda x: self.__get_polarity(x))

        nd.loc[nd.polarity >= 0.36, 'sentiment'] = 'positive'
        nd.loc[((nd.polarity > 0.0) & (nd.polarity < 0.36)), 'sentiment'] = 'neutral'
        nd.loc[nd.polarity < 0, 'sentiment'] = 'negative'
        df1 = nd[(nd.sentiment != "negativo") & (nd.sentiment == "positive")][0:430]
        df2 = nd[(nd.sentiment == "negative")][0:430]
        df3 = nd[(nd.sentiment == "neutral")][0:430]

        return pd.concat([df1, df2, df3])

    def pre_cleaning(self) -> None:
        dataframe = pd.read_csv(f"{self.path}/original/FoodReviews.csv")
        dataframe = self.sample_df(dataframe)
        self.write_dataframe(dataframe, f"{self.path}/translated/FoodReviewsPreClean.xlsx")

    @property
    def __columns(self) -> List:
        return ["id_review", "review", "clean_review", "sentiment", "words_len", "polarity"]

    def process(self) -> None:
        dataset = self.read_dataframe(f"{self.path}/translated/food_reviews_es.xlsx")
        dataset = dataset.sort_values(by=['sentiment'])[self.__columns]
        self.write_dataframe(dataset, f"{self.path}/processed/food_reviews_processed.xlsx")
