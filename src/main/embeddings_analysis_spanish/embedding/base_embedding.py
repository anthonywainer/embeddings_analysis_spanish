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
        gpt2 = "datificate/gpt2-small-spanish"

        self.gpt2_model = TFGPT2Model.from_pretrained(gpt2)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2, do_lower_case=True)

        bert = "dccuchile/bert-base-spanish-wwm-uncased"
        self.bert_model = TFBertModel.from_pretrained(bert)
        self.beto_tokenizer = BertTokenizer.from_pretrained(bert, do_lower_case=True)

        self.w2b_es = KeyedVectors.load_word2vec_format(f'{self.gensim_path}/SBW-vectors-300-min5.txt')
        self.ft_es = KeyedVectors.load_word2vec_format(f'{self.gensim_path}/embeddings-l-model.vec')
        self.glove_es = KeyedVectors.load_word2vec_format(f'{self.gensim_path}/glove-sbwc.i25.vec')

    @staticmethod
    def extract_gpt_embedding(model: TFGPT2Model.from_pretrained,
                              texts: np.array,
                              tokenizer: GPT2Tokenizer.from_pretrained) -> np.ndarray:
        """
        Method to extract embedding from GPT2
        :param model: GPT2 Model from TensorFlow
        :param texts: Text in format array numpy
        :param tokenizer: GPT2 Tokenizer from TensorFlow
        :return: dimensional array with embeddings
        """

        max_len: int = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            tokenizer.pad_token = "[PAD]"
            pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
            features = pipe(text)
            features = np.squeeze(features)
            _array[idx] = features.mean(axis=0) if len(features) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_beto_embedding(model: TFBertModel.from_pretrained,
                               texts: np.array,
                               tokenizer: BertTokenizer.from_pretrained) -> np.ndarray:
        """
        Method to extract embedding from Bert
        :param model: Bert Model from TensorFlow
        :param texts: Text in format array numpy
        :param tokenizer: Bert Tokenizer from TensorFlow
        :return: dimensional array with embeddings
        """
        max_len = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            input_ids = tf.constant(tokenizer.encode(text))[None, :]
            outputs = model(input_ids)
            last_hidden_states = outputs["pooler_output"]
            cls_token = last_hidden_states[0]

            _array[idx] = np.mean(cls_token, axis=0) if len(cls_token) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_embedding(model: KeyedVectors.load_word2vec_format,
                          words: List, max_len: int) -> np.ndarray:
        """
        Method to extract embedding from KeyedVectors
        :param model: KeyedVectors Model
        :param words: Words to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """
        data = np.array([model.wv[w] for w in words if w in model.wv.index2word])

        return np.mean(data, axis=0) if len(data) else np.zeros(max_len)

    def extract_gensim_embedding(self, model: KeyedVectors.load_word2vec_format,
                                 texts: np.array, max_len: int) -> np.ndarray:
        """
        Method to extract embedding from KeyedVectors
        :param model: KeyedVectors Model
        :param texts: Text to extract embeddings
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """

        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            _array[idx] = self.extract_embedding(model, text.split(), max_len)

        return StandardScaler().fit_transform(_array)

    @property
    def embeddings(self) -> List:
        return ["gpt2", "bert", "w2v", "fastText", "glove"]

    def extract(self, name: str, values: np.array, max_len) -> np.ndarray:
        """
        Method to extract embedding from dict
        :param name: Name to extract
        :param values: Words to process
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """
        return LazyDict({
            "gpt2": (self.extract_gpt_embedding, (self.gpt2_model, values, self.gpt2_tokenizer)),
            "bert": (self.extract_beto_embedding, (self.bert_model, values, self.beto_tokenizer)),
            "w2v": (self.extract_gensim_embedding, (self.w2b_es, values, max_len)),
            "fastText": (self.extract_gensim_embedding, (self.ft_es, values, max_len)),
            "glove": (self.extract_gensim_embedding, (self.glove_es, values, max_len))
        }).get(name)

    def extract_array(self, name: str, x_: np.array, max_len: int = 300) -> np.ndarray:
        """
        Method to load or save array
        :param name: Name to extract
        :param x_: Words to process
        :param max_len: Max length to create dimension
        :return: dimensional array with embeddings
        """

        if not os.path.exists(f"{self.numpy_path}/{name}.npz"):
            vec = self.extract(name, x_.values, max_len)
            np.savez(f"{self.numpy_path}/{name}", vec)
            self.logger.info("saved successfully", f"{self.numpy_path}/{name}.npz")
            return vec
        else:
            self.logger.info("loaded successfully")
            return np.load(f"{self.numpy_path}/{name}.npz")["arr_0"]
