from typing import List

import numpy as np
from gensim.models import KeyedVectors

from sklearn.preprocessing import StandardScaler

from embeddings_analysis_spanish.embeddings.base_embedding import BaseEmbedding


class GensimEmbedding(BaseEmbedding):
    """
    Based in gensim @author: dccuchile
    * W2V
    * FASTTEXT
    * GLOVE
    """

    def __init__(self) -> None:
        """
        Init embeddings extraction
        """

        super().__init__()
        self.locals = locals()

    @staticmethod
    def process_gensim_embedding(model: KeyedVectors.load_word2vec_format, words: List, max_len: int) -> np.ndarray:
        """
        Method to process embeddings from KeyedVectors
        :param model: KeyedVectors Model
        :param words: Words to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """
        data = np.array([model[w] for w in words if w in model.index_to_key])

        return np.mean(data, axis=0) if len(data) else np.zeros(max_len)

    def extract_gensim_embedding(self, embedding_name: str, texts: np.array, max_len: int) -> np.ndarray:
        """
        Method to extract embeddings from KeyedVectors
        :param embedding_name: Name to extract
        :param texts: Text to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """

        if embedding_name not in self.locals:
            self.logger.info(f"Loading {self.gensim_path}/{embedding_name}.vec")
            self.locals[embedding_name] = KeyedVectors.load_word2vec_format(f'{self.gensim_path}/{embedding_name}.vec')

        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            _array[idx] = self.process_gensim_embedding(self.locals[embedding_name], text.split(), max_len)

        return StandardScaler().fit_transform(_array)
