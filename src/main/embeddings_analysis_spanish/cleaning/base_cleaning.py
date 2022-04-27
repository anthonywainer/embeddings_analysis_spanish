from typing import Any, AnyStr

import pandas as pd

from embeddings_analysis_spanish.utils.logger import Logger
from embeddings_analysis_spanish.utils.wrapper_path import is_path


class BaseCleaning(Logger):

    def __init__(self, path: str = "data/dataset") -> None:
        super().__init__()
        self.path = path

    @staticmethod
    def _count_words(text: Any) -> int:
        """
        Count words very easy, valid first None and after string
        @param text: Text to process
        @return: 0 if is None
        """
        if (text is not None) and (isinstance(text, str)):
            words = set(text.split())
            if words:
                return len(words)
        return 0

    @is_path
    def read_dataframe(self, path: AnyStr) -> pd.DataFrame:
        """
        Read excel from valid path
        :param path: The path
        :return: Dataframe
        """
        self.logger.info(f"Reading excel from {self.path}")
        return pd.read_excel(path)

    @staticmethod
    def write_dataframe(dataframe: pd.DataFrame, path: AnyStr) -> None:
        dataframe.to_excel(path, index=False)
