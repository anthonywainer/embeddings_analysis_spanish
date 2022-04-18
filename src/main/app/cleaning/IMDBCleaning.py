import pandas as pd

from .BaseCleaning import BaseCleaning
from ..utils.cleaner import processing_words


class IMDBCleaning(BaseCleaning):

    def pre_cleaning(self) -> None:
        import cld3
        dataset = pd.read_csv(f"{self.path}/original/IMDB Dataset SPANISH.csv")
        dataset.loc[:, ('clean_review')] = dataset.review_es.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='clean_review', keep="last")
        dataset.sort_index(inplace=True)

        dataset = dataset[dataset.review_es.apply(
            lambda x: cld3.get_language(x).language != "en"
        )][["review_es", "sentimiento", "clean_review"]]

        dataset = dataset.rename(columns={"clean_review_es": "clean_review", "sentimiento": "sentiment"})
        dataset.to_excel(f"{self.path}/translated/IMDBPreClean.xlsx", index=False)

    @property
    def __columns(self) -> List:
        return ["id", "review", "clean_review", "sentiment", "words_len"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{self.path}/translated/IMDBES.xlsx")
        dataset["words_len"] = dataset.clean_review.apply(lambda c: len(set(c.split())))
        dataset["id"] = range(0, len(dataset))

        positive_df = dataset[(dataset.words_len >= 200) & (dataset.sentiment == "positivo")][0:1500]
        negative_df = dataset[(dataset.words_len >= 200) & (dataset.sentiment == "negativo")][0:1500]

        dataset = pd.concat([positive_df, negative_df])[self.__columns]

        dataset = dataset.sort_values(by=['sentiment'])
        dataset.to_excel(f"{self.path}/processed/IMDBProcessed.xlsx", index=False)
