import os
from typing import List

import numpy as np
import tensorflow as tf

from transformers import TFGPT2Model, GPT2Tokenizer, BertTokenizer, TFBertModel
from transformers import pipeline
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

from embeddings_analysis_spanish.utils.logger import Logger
from embeddings_analysis_spanish.utils.mapping import LazyDict


class BaseEmbedding(Logger):
    """
    Base Embedding support the next models in spanish:
    * GPT2 - @author: datificate/gpt2-small-spanish
    * BETO - @author: dccuchile/bert-base-spanish-wwm-uncased

    Based in gensim @author: dccuchile
    * W2V
    * FASTTEXT
    * GLOVE
    """

    def __init__(self, gensim_path: str = "data/gensim", numpy_path: str = "data/numpy") -> None:
        """
        Init embedding extraction
        :param gensim_path: Path where is vectors Gensim
        :param numpy_path: Path to save or load vector numpy
        """

        super().__init__()
        self.gensim_path = gensim_path
        self.numpy_path = numpy_path

        self.gpt2_name = "datificate/gpt2-small-spanish"
        self.gpt2_model = None
        self.gpt2_tokenizer = None

        self.bert_name = "dccuchile/bert-base-spanish-wwm-uncased"
        self.bert_model = None
        self.bert_tokenizer = None

        locals()["w2v"] = None
        locals()["fast_text"] = None
        locals()["glove"] = None
        self.locals = locals()

    def extract_gpt_embedding(self, texts: np.array) -> np.ndarray:
        """
        Method to extract embedding from GPT2
        :param texts: Text in format array numpy
        :return: dimensional array with embeddings
        """

        if self.bert_model is None:
            self.bert_model = TFBertModel.from_pretrained(self.bert_name)
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_name, do_lower_case=True)

        max_len: int = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            self.bert_tokenizer.pad_token = "[PAD]"
            pipe = pipeline('feature-extraction', model=self.bert_model, tokenizer=self.bert_tokenizer)
            features = pipe(text)
            features = np.squeeze(features)
            _array[idx] = features.mean(axis=0) if len(features) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    def extract_beto_embedding(self, texts: np.array) -> np.ndarray:
        """
        Method to extract embedding from Bert - Beto
        :param texts: Text in format array numpy
        :return: dimensional array with embeddings
        """
        if self.gpt2_model is None:
            self.gpt2_model = TFGPT2Model.from_pretrained(self.gpt2_name)
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_name, do_lower_case=True)

        max_len = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            input_ids = tf.constant(self.gpt2_tokenizer.encode(text))[None, :]
            outputs = self.gpt2_model(input_ids)
            last_hidden_states = outputs["pooler_output"]
            cls_token = last_hidden_states[0]

            _array[idx] = np.mean(cls_token, axis=0) if len(cls_token) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def process_embedding(model: KeyedVectors.load_word2vec_format,
                          words: List, max_len: int) -> np.ndarray:
        """
        Method to process embedding from KeyedVectors
        :param model: KeyedVectors Model
        :param words: Words to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """
        data = np.array([model.wv[w] for w in words if w in model.wv.index2word])

        return np.mean(data, axis=0) if len(data) else np.zeros(max_len)

    def extract_gensim_embedding(self, embedding_name: str, texts: np.array, max_len: int) -> np.ndarray:
        """
        Method to extract embedding from KeyedVectors
        :param embedding_name: Name to extract
        :param texts: Text to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """

        if self.locals[embedding_name] is None:
            self.logger.info(f"Loading {self.gensim_path}/{embedding_name}.vec")
            self.locals[embedding_name] = KeyedVectors.load_word2vec_format(f'{self.gensim_path}/{embedding_name}.vec')

        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            _array[idx] = self.process_embedding(self.locals[embedding_name], text.split(), max_len)

        return StandardScaler().fit_transform(_array)

    @property
    def embeddings(self) -> List:
        return ["gpt2", "bert", "w2v", "fast_text", "glove"]

    def extract(self, embedding_name: str, values: np.array, max_len: int) -> np.ndarray:
        """
        Method to extract embedding from dict
        :param embedding_name: Name to extract
        :param values: Words to process
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """
        return LazyDict({
            "gpt2": (self.extract_gpt_embedding, (values,)),
            "bert": (self.extract_beto_embedding, (values,)),
            "w2v": (self.extract_gensim_embedding, (embedding_name, values, max_len)),
            "fast_text": (self.extract_gensim_embedding, (embedding_name, values, max_len)),
            "glove": (self.extract_gensim_embedding, (embedding_name, values, max_len))
        }).get(embedding_name)

    def extract_embedding(self, embedding_name: str, dataset_name: str, x_: np.array, max_len: int = 300) -> np.ndarray:
        """
        Method to load or save array with embeddings
        :param embedding_name: Name to extract
        :param dataset_name: Dataset name to process
        :param x_: Words to process
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """

        if not os.path.exists(f"{self.numpy_path}/{dataset_name}/{embedding_name}.npz"):
            vec = self.extract(embedding_name, x_.values, max_len)
            np.savez(f"{self.numpy_path}/{dataset_name}/{embedding_name}", vec)
            self.logger.info(f"saved successfully - {self.numpy_path}/{dataset_name}/{embedding_name}.npz")
            return vec
        else:
            self.logger.info("loaded successfully")
            return np.load(f"{self.numpy_path}/{dataset_name}/{embedding_name}.npz", allow_pickle=True)["arr_0"]
