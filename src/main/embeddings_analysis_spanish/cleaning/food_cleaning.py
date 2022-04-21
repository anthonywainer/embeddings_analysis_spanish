from typing import List

import pandas as pd
from textblob import TextBlob

from .base_cleaning import BaseCleaning
from ..utils.cleaner import processing_words


class FoodCleaning(BaseCleaning):

    @staticmethod
    def __get_polarity(text: str) -> str:
        try:
            t = TextBlob(text)
            return t.translate(from_lang="es", to="en").correct().sentiment.polarity
        except Exception as e:
            print("ERROR", e)
            return ""

    def pre_cleaning(self) -> None:
        dataset = pd.read_csv(f"{self.path}/original/FoodReviews.csv")

        dataset.loc[:, 'clean_review'] = dataset.review.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='review', keep="last")
        dataset.sort_index(inplace=True)
        negative = dataset[dataset.score.astype("int") < 3][:57000]
        negative["sentiment"] = "negativo"
        neutro = dataset[dataset.score.astype("int") == 3][:57000]
        neutro["sentiment"] = "neutro"
        positive = dataset[dataset.score.astype("int") > 3][:57000]
        positive["sentiment"] = "positivo"
        dataset = pd.concat([negative, neutro, positive])[["id_review", "review", "sentiment", "clean_review"]]

        dataset["words_len"] = dataset.clean_review.apply(lambda c: self._count_words(c))
        nd = dataset[dataset.words_len >= 60]
        nd['Polarity'] = nd['review'].apply(lambda x: self.__get_polarity(x))

        nd.loc[nd.Polarity >= 0.36, 'sentiment'] = 'positive'
        nd.loc[((nd.Polarity > 0.0) & (nd.Polarity < 0.36)), 'sentiment'] = 'neutral'
        nd.loc[nd.Polarity < 0, 'sentiment'] = 'negative'
        df1 = nd[(nd.sentiment != "negativo") & (nd.sentiment == "positive")][0:430]
        df2 = nd[(nd.Sentiment == "negative")][0:430]
        df3 = nd[(nd.Sentiment == "neutral")][0:430]

        new_dataset = pd.concat([df1, df2, df3])

        new_dataset.to_excel(f"{self.path}/translated/FoodReviewsPreClean.xlsx", index=False)

    @property
    def __columns(self) -> List:
        return ["id_review", "review", "clean_review", "sentiment", "words_len", "polarity"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{self.path}/translated/food_reviews_es.xlsx")
        dataset = dataset.sort_values(by=['sentiment'])[self.__columns]
        dataset.to_excel(f"{self.path}/processed/food_reviews_processed.xlsx", index=False)
