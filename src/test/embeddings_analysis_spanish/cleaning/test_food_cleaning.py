import pandas as pd
from typing import AnyStr

import pytest

from embeddings_analysis_spanish.cleaning.food_cleaning import FoodCleaning


class FoodCleaningMock(FoodCleaning):
    def read_dataframe(self, path: AnyStr) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["R129248",
                 """
                    No hagan caso a los camareros, pues tienen la costumbre de aconsejarte lo mas caro y peor calidad. 
                    En cuanto a las bebidas, la praxis que tienen los camareros es miserable. 
                    Siempre intentaran colocarte la bebida mas cara de la bodega y si no andas 
                    con cuidado no podras pagar la cuenta.
                    Sobre todo con los alcoholes de la sobremesa.                 
                 """, 3],
                ["R77336",
                 """
                    Las dos veces que he ido me han hecho la misma jugada y por lo que veo en los comentarios es                 
                 """, 5]
            ],
            columns=["id_review", "review", "score"]
        )

    @staticmethod
    def write_dataframe(dataframe: pd.DataFrame, path: AnyStr) -> None:
        pass


@pytest.fixture
def food_cleaning():
    return FoodCleaningMock()


def test_sample_df(imdb_cleaning):
    dataframe = imdb_cleaning.read_dataframe("dummy.xlsx")
    result = imdb_cleaning.sample_df(dataframe)

    assert len(result) == 0
    assert "clean_review" in result.columns


def test_process(imdb_cleaning):
    assert imdb_cleaning.process() is None
