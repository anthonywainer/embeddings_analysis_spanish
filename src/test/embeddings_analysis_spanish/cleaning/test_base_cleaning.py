from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning


def test__count_words():
    base_cleaning = BaseCleaning()

    assert base_cleaning._count_words("dummy") == 1


def test__count_words_none():
    base_cleaning = BaseCleaning()

    assert base_cleaning._count_words(None) == 0


def test__count_words_type():
    base_cleaning = BaseCleaning()

    assert base_cleaning._count_words(1) == 0
