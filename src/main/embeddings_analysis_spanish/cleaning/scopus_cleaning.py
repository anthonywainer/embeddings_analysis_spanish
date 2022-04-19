from typing import List

import pandas as pd

from .base_cleaning import BaseCleaning
from ..utils.cleaner import processing_words


class ScopusCleaning(BaseCleaning):

    @property
    def __features(self) -> List:
        return ["agricultura", "cultura", "deporte",
                "economia", "salud", "tecnologia"]

    def process(self) -> None:
        dataset = pd.read_excel(f"{self.path}/translated/PaperScopusES.xlsx")

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
        dataset.to_excel(f"{self.path}/processed/PaperScopusProcessed.xlsx", index=False)
