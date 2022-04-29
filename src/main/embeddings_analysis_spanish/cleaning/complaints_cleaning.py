from typing import List

import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


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
        """
        List to remove string as XX XXX XXXX
        :return: XX XXX XXXX
        """
        return ["x" * x for x in range(4, 20)]

    def sample_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[:, 'clean_complaints_en'] = dataframe["Consumer Complaint"].apply(
            lambda d: processing_words(d, self.__own_stop_words, lang="english")
        )

        dataframe["words_len"] = dataframe["clean_complaints_en"].apply(lambda c: self._count_words(c))
        return pd.concat(
            map(
                lambda f: dataframe[dataframe.words_len >= 50][
                              (dataframe[dataframe.words_len >= 50].Product == f)
                          ][:1000],
                self.__features
            )
        ).rename(columns={
            "Complaint ID": "id",
            "Consumer Complaint": "complaint",
            "Product": "product"
        })

    def pre_cleaning(self) -> None:
        """
        Building Sample > 50 words length
        """
        dataframe = self.read_dataframe(f"{self.path}/original/ConsumerComplaints.xlsx")
        dataframe = self.sample_df(dataframe)
        self.write_dataframe(
            dataframe.sort_values(by=['Product']),
            f"{self.path}/translated/ConsumerComplaintsPreClean.xlsx"
        )

    def cooking_process(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[:, 'clean_complaints'] = dataframe["complaint"].apply(
            lambda d: processing_words(d, self.__own_stop_words)
        )
        return dataframe.sort_values(
            by=['product']
        )

    def process(self) -> None:
        dataframe = pd.read_excel(f"{self.path}/translated/complaints_es.xlsx")
        dataframe = self.cooking_process(dataframe)

        self.write_dataframe(dataframe, f"{self.path}/processed/complaints_processed.xlsx")
