from abc import ABC, abstractmethod
from typing import Tuple, Union

import pandas as pd


class AbstractModel(ABC):

    def _show_result(self, data) -> pd.DataFrame: ...

    @staticmethod
    def __save_data(path, predicted_embedding: Union) -> None: ...

    def fit_predict(self, **kwargs) -> Tuple: raise NotImplementedError("this method should implement!")

    @abstractmethod
    def run(self, **kwargs) -> Tuple: raise NotImplementedError("this method should implement!")
