import pandas as pd
from typing import AnyStr

import pytest

from embeddings_analysis_spanish.cleaning.bbc_cleaning import BBCCleaning


class BBCCleaningMock(BBCCleaning):
    def read_dataframe(self, path: AnyStr) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["negocio", "Ventas de anuncios Boost Time Warner ganancias",
                 """Las ganancias trimestrales en el gigante de los medios de comunicación de EE. UU. """],
                ["negocio", "Ganancias de dólar en el discurso de Greenspan",
                 """El dólar ha alcanzado su nivel más alto contra el euro en casi tres meses """],
                ["negocio", "Los precios de los altos combustibles golpean las ganancias de BA",
                 """Los propietarios de Yukos gigantes de petróleo ruso asediados"""],
                ["negocio", "Los precios de los altos combustibles golpean las ganancias de BA",
                 """Los propietarios de Yukos gigantes de petróleo ruso asediados"""]
            ],
            columns=["category", "title", "content"]
        )

    @staticmethod
    def write_dataframe(dataframe: pd.DataFrame, path: AnyStr) -> None:
        pass


@pytest.fixture
def bbc_cleaning():
    return BBCCleaningMock()


def test_sample_df(imdb_cleaning):
    dataframe = imdb_cleaning.read_dataframe("dummy.xlsx")
    result = imdb_cleaning.sample_df(dataframe)

    assert len(result) == 3
    assert "clean_news" in result.columns
    assert result["clean_news"][0].split().sort() == "comunicación medios gigante ganancias trimestrales".split().sort()


def test_process(imdb_cleaning):
    assert imdb_cleaning.process() is None
