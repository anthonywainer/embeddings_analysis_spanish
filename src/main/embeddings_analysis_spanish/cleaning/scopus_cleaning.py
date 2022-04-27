from typing import List

import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class ScopusCleaning(BaseCleaning):

    @property
    def __features(self) -> List:
        return ["agricultura", "cultura", "deporte",
                "economia", "salud", "tecnologia"]

    def process(self) -> None:
        """
        Building Sample > 50 words length
        """
        dataset = self.read_dataframe(f"{self.path}/translated/paper_scopus_es.xlsx")

        dataset.loc[:, 'clean_abstract'] = dataset.abstract.apply(processing_words)
        dataset = dataset.drop_duplicates(subset='DOI', keep="last")

        dataset["words_len"] = dataset["abstracts"].apply(lambda c: self._count_words(c))

        dataset = pd.concat(
            map(
                lambda f: dataset[dataset.words_len >= 90][(dataset[dataset.words_len >= 90].category == f)][:500],
                self.__features
            )
        )

        dataset = dataset.sort_values(by=['category'])
        self.write_dataframe(dataset, f"{self.path}/processed/paper_scopus_processed.xlsx")
