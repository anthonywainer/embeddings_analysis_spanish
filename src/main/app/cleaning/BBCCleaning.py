import pandas as pd

from .BaseCleaning import BaseCleaning
from ..utils.cleaner import processing_words


class BBCCleaning(BaseCleaning):

    def process(self) -> None:
        dataset = pd.read_excel(f"{self.path}/translated/BBCNewsES.xlsx")

        dataset.loc[:, 'clean_news'] = dataset.content.apply(processing_words)
        dataset["id"] = range(0, len(dataset))

        dataset = dataset.drop_duplicates(subset='clean_news', keep="last")
        dataset = dataset.sort_values(by=["category"])

        dataset.to_excel(f"{self.path}/processed/BBCNewsProcessed.xlsx", index=False)
