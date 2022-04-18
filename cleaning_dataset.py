from typing import Any, List

from textblob import TextBlob

from utils.cleaner import processing_words
import pandas as pd

MAIN_PATH = "dataset"


class BaseCleaning(object):
    @staticmethod
    def _count_words(text: Any) -> int:
        """
        Count words very easy, valid first is None and if is string
        @param text: Text to process
        @return: 0 if is None
        """
        if (text is not None) and (isinstance(text, str)):
            words = set(text.split())
            if words:
                return len(words)
        return 0


class BBCCleaning(object):
    @staticmethod
    def process() -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/translated/BBCNewsES.xlsx")

        dataset.loc[:, 'clean_news'] = dataset.content.apply(processing_words)
        dataset["id"] = range(0, len(dataset))

        dataset = dataset.drop_duplicates(subset='clean_news', keep="last")
        dataset = dataset.sort_values(by=["category"])

        dataset.to_excel(f"{MAIN_PATH}/processed/BBCNewsProcessed.xlsx", index=False)


class ComplaintsCleaning(BaseCleaning):

    @property
    def __features(self) -> List:
        return ["Bank account or service", "Checking or savings account",
                "Consumer Loan", "Credit card", "Credit reporting",
                "Debt collection", "Money transfers",
                "Mortgage", "Payday loan", "Prepaid card",
                "Student loan", "Vehicle loan or lease"]

    @property
    def __own_stop_words(self) -> List:
        return ["x" * x for x in range(4, 20)]

    def pre_cleaning(self) -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/original/ConsumerComplaints.xlsx")

        dataset.loc[:, 'clean_complaints'] = dataset["Consumer Complaint"].apply(
            lambda d: processing_words(d, self.__own_stop_words, lang="english")
        )

        dataset["words_len"] = dataset["clean_complaints"].apply(lambda c: self._count_words(c))
        dataset = dataset.drop_duplicates(subset='clean_complaints', keep="last")
        dataset = pd.concat(
            map(
                lambda f: dataset[dataset.words_len >= 50][(dataset[dataset.words_len >= 50].Product == f)][:1000],
                self.__features
            )
        )
        dataset.sort_values(
            by=['Product']
        ).to_excel(f"{MAIN_PATH}/translated/ConsumerComplaintsPreClean.xlsx", index=False)

    def process(self) -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/translated/ConsumerComplaintsEs.xlsx")

        dataset.loc[:, 'clean_complaints'] = dataset["complaint"].apply(
            lambda d: processing_words(d, self.__own_stop_words)
        )
        dataset = dataset.sort_values(
            by=['product']
        )
        dataset.to_excel(f"{MAIN_PATH}/processed/ConsumerComplaintsProcessed.xlsx", index=False)


class IMDBCleaning(object):

    @staticmethod
    def pre_cleaning() -> None:
        dataset = pd.read_csv(f"{MAIN_PATH}/original/IMDB Dataset SPANISH.csv")
        dataset.loc[:, ('clean_review')] = dataset.review_es.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='clean_review', keep="last")
        dataset.sort_index(inplace=True)

        dataset = dataset[dataset.review_es.apply(
            lambda x: cld3.get_language(x).language != "en"
        )][["review_es", "sentimiento", "clean_review"]]

        dataset = dataset.rename(columns={"clean_review_es": "clean_review", "sentimiento": "sentiment"})
        dataset.to_excel(f"{MAIN_PATH}/translated/IMDBPreClean.xlsx", index=False)

    @property
    def __columns(self) -> List:
        return ["id", "review", "clean_review", "sentiment", "words_len"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/translated/IMDBES.xlsx")
        dataset["words_len"] = dataset.clean_review.apply(lambda c: len(set(c.split())))
        dataset["id"] = range(0, len(dataset))

        positive_df = dataset[(dataset.words_len >= 200) & (dataset.sentiment == "positivo")][0:1500]
        negative_df = dataset[(dataset.words_len >= 200) & (dataset.sentiment == "negativo")][0:1500]

        dataset = pd.concat([positive_df, negative_df])[self.__columns]

        dataset = dataset.sort_values(by=['sentiment'])
        dataset.to_excel(f"{MAIN_PATH}/processed/IMDBProcessed.xlsx", index=False)


class ScopusCleaning(BaseCleaning):

    @property
    def __features(self) -> List:
        return ["agricultura", "cultura", "deporte",
                "economia", "salud", "tecnologia"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/translated/PaperScopusES.xlsx")

        dataset.loc[:, 'clean_abstract'] = dataset.abstract.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='DOI', keep="last")

        dataset["words_len"] = dataset["abstract"].apply(lambda c: self._count_words(c))

        dataset = pd.concat(
            map(
                lambda f: dataset[dataset.words_len >= 90][(dataset[dataset.words_len >= 90].category == f)][:500],
                self.__features
            )
        )

        dataset = dataset.sort_values(by=['category'])
        dataset.to_excel(f"{MAIN_PATH}/processed/PaperScopusProcessed.xlsx", index=False)


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
        dataset = pd.read_csv(f"{MAIN_PATH}/original/FoodReviews.csv")

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
        df1 = nd[(nd.sentimiento != "negativo") & (nd.Sentiment == "positive")][0:430]
        df2 = nd[(nd.Sentiment == "negative")][0:430]
        df3 = nd[(nd.Sentiment == "neutral")][0:430]

        new_dataset = pd.concat([df1, df2, df3])

        new_dataset.to_excel(f"{MAIN_PATH}/translated/FoodReviewsPreClean.xlsx", index=False)

    @property
    def __columns(self) -> List:
        return ["id_review", "review", "clean_review", "sentiment", "words_len", "polarity"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{MAIN_PATH}/translated/FoodReviewsES.xlsx")
        dataset = dataset.sort_values(by=['sentiment'])[self.__columns]
        dataset.to_excel(f"{MAIN_PATH}/processed/FoodReviewsProcessed.xlsx", index=False)


"""
if __name__ == "__main__":
    BBCCleaning().process()
    ComplaintsCleaning().process()
    IMDBCleaning().process()
    ScopusCleaning().process()
    FoodCleaning().process()
"""
