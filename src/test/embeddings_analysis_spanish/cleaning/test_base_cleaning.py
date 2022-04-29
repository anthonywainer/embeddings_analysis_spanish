from pathlib import Path

import os
import pytest
import embeddings_analysis_spanish
import unittest.mock as mock
import pandas as pd

from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning, is_path


@pytest.fixture
def base_cleaning():
    return BaseCleaning()


def test__count_words(base_cleaning):
    assert base_cleaning._count_words("dummy") == 1


def test__count_words_none(base_cleaning):
    assert base_cleaning._count_words(None) == 0


def test__count_words_type(base_cleaning):
    assert base_cleaning._count_words(1) == 0


def test_is_path():
    def mock_path(path_dum: str) -> str:
        return path_dum

    path = f"{os.path.dirname(embeddings_analysis_spanish.__file__)}/cleaning/bbc_cleaning.py"
    assert is_path(mock_path)(path) == Path(path)


@mock.patch('embeddings_analysis_spanish.utils.wrapper_path.is_path')
def test_read_dataframe(base_cleaning):
    with mock.patch.object(pd, "read_excel") as read_excel_mock:
        base_cleaning.read_dataframe("Stuff.xlsx")
        read_excel_mock.assert_not_called()


def test_write_dataframe(base_cleaning):
    test_df = pd.DataFrame(
        [],
        columns=["A", "B"]
    )

    with mock.patch.object(test_df, "to_excel") as to_excel_mock:
        base_cleaning.write_dataframe(test_df, "Stuff.xlsx")
        to_excel_mock.assert_called_with('Stuff.xlsx', index=False)
