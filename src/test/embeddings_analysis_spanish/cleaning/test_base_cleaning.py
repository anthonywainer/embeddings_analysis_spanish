from pathlib import Path

import os
import pytest
import embeddings_analysis_spanish

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
