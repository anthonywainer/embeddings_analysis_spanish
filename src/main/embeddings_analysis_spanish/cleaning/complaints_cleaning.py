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

    def pre_cleaning(self) -> None:
        """
        Building Sample > 50 words length
        """
        dataset = self.read_dataframe(f"{self.path}/original/ConsumerComplaints.xlsx")

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
        self.write_dataframe(
            dataset.sort_values(by=['Product']),
            f"{self.path}/translated/ConsumerComplaintsPreClean.xlsx"
        )

    def process(self) -> None:
        dataset = pd.read_excel(f"{self.path}/translated/complaints_es.xlsx")

        dataset.loc[:, 'clean_complaints'] = dataset["complaint"].apply(
            lambda d: processing_words(d, self.__own_stop_words)
        )
        dataset = dataset.sort_values(
            by=['product']
        )
        self.write_dataframe(dataset, f"{self.path}/processed/complaints_processed.xlsx")
