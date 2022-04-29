import pandas as pd
from typing import AnyStr

import pytest

from embeddings_analysis_spanish.cleaning.scopus_cleaning import ScopusCleaning


class ScopusCleaningMock(ScopusCleaning):
    def read_dataframe(self, path: AnyStr) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["agricultura",
                 """
                    En este documento, proponemos analizar el programa para la expansión y la mejora de la educación 
                    rural (EMER) en la provincia de Entre Riós entre 1978 y 1992. Es un programa que propone un vínculo 
                    entre la educación formal y las competencias laborales en las escuelas primarias rurales a través 
                    del Incorporación de talleres de trabajo manuales, artes marítimos y agricultura, viable a través
                     de la nuclearización de las escuelas y la regionalización de los contenidos. Ha sido financiado 
                     por el Banco Interamericano de Desarrollo, con una contraparte equivalente de la Tesorería Nacional
                      y se llevó a cabo en todo el país, con diferentes tasas y niveles de aplicación. Participamos 
                      en el debate sobre las influencias externas de las organizaciones internacionales de 
                      financiamiento sobre políticas educativas a nivel nacional y regional. Nuestra hipótesis 
                      es que, en sus orígenes, este programa tuvo una correlación con las políticas de 
                      descentralización de la última dictadura y que continuó durante la década de 1980. 
                      Concluimos que el Programa EMER fue un elemento legitimante de la política de transferir 
                      escuelas primarias nacionales a las provincias y el alcance principal de las políticas de la 
                  """, "123sae"],
                ["cultura",
                 """
                    La reciente muerte de Miguel Artola (1923-2020) presenta la oportunidad de reflexionar sobre su 
                    carrera como historiador. En el difícil contexto de España, bajo Franco, Artola representó una 
                    tercera forma de historiografía liberal. Se esforzó por recuperar el pasado olvidado y rico del 
                 """, "123s2ae"]
            ],
            columns=["category", "abstract", "DOI"]
        )

    @staticmethod
    def write_dataframe(dataframe: pd.DataFrame, path: AnyStr) -> None:
        pass


@pytest.fixture
def scopus_cleaning():
    return ScopusCleaningMock()


def test_sample_df(scopus_cleaning):
    dataframe = scopus_cleaning.read_dataframe("dummy.xlsx")
    result = scopus_cleaning.sample_df(dataframe)

    assert len(result) == 1
    assert "clean_abstract" in result.columns


def test_process(scopus_cleaning):
    assert scopus_cleaning.process() is None
