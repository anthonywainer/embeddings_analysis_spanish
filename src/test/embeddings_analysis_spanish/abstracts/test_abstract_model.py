from typing import Tuple

import pytest

from main.embeddings_analysis_spanish.abstracts.abstract_model import AbstractModel


def test_abstract_class():
    with pytest.raises(TypeError) as error:
        abstract_model = AbstractModel()
        raise abstract_model._show_result(data=iter([]))

    assert str(error.value) == "Can't instantiate abstract class AbstractModel with abstract methods run"


def test_fit_predict():
    with pytest.raises(NotImplementedError) as error:
        class AbstractModelDummy(AbstractModel):
            def run(self, **kwargs) -> Tuple:
                pass

        raise AbstractModelDummy().fit_predict()

    assert str(error.value) == "this method should implement!"


def test_run():
    with pytest.raises(BaseException) as error:
        class AbstractModelDummy(AbstractModel):

            def fit_predict(self, **kwargs) -> Tuple:
                pass

        raise AbstractModelDummy().run()

    assert str(error.value) == "this method should implement!"
