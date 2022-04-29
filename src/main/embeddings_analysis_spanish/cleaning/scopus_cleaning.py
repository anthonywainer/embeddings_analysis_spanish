from typing import List

import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class ScopusCleaning(BaseCleaning):

    @property
    def __features(self) -> List:
        return ["agricultura", "cultura", "deporte",
                "economia", "salud", "tecnologia"]

    def sample_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[:, 'clean_abstract'] = dataframe.abstract.apply(processing_words)
        dataframe = dataframe.drop_duplicates(subset='DOI', keep="last")

        dataframe["words_len"] = dataframe["abstract"].apply(lambda c: self._count_words(c))

        dataframe = pd.concat(
            map(
                lambda f: dataframe[dataframe.words_len >= 90][
                              (dataframe[dataframe.words_len >= 90].category == f)
                          ][:500],
                self.__features
            )
        )

        return dataframe.sort_values(by=['category'])

    def process(self) -> None:
        """
        Building Sample > 50 words length
        """
        dataframe = self.read_dataframe(f"{self.path}/translated/paper_scopus_es.xlsx")
        dataframe = self.sample_df(dataframe)
        self.write_dataframe(dataframe, f"{self.path}/processed/paper_scopus_processed.xlsx")
