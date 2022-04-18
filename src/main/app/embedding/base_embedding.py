import os
from typing import List

import numpy as np
import tensorflow as tf

from transformers import TFGPT2Model, GPT2Tokenizer, BertTokenizer, TFBertModel
from transformers import pipeline
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

from app.utils.logger import Logger


class BaseEmbedding(Logger):

    def __init__(self, gensim_path: str = "data/gensim", numpy_path: str = "data/numpy") -> None:
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
    def extract_gpt_embedding(texts: np.array, tokenizer,
                              model: TFGPT2Model.from_pretrained,
                              max_len: int = 768) -> np.ndarray:
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            tokenizer.pad_token = "[PAD]"
            pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
            features = pipe(text)
            features = np.squeeze(features)
            _array[idx] = features.mean(axis=0) if len(features) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_beto_embedding(model: TFBertModel.from_pretrained, tokenizer, texts: np.array,
                               max_len: int = 768) -> np.ndarray:
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            input_ids = tf.constant(tokenizer.encode(text))[None, :]
            outputs = model(input_ids)
            last_hidden_states = outputs["pooler_output"]
            cls_token = last_hidden_states[0]

            _array[idx] = np.mean(cls_token, axis=0) if len(cls_token) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_embedding(model: KeyedVectors.load_word2vec_format, words: List, max_len: int) -> np.ndarray:
        data = np.array([model.wv[w] for w in words if w in model.wv.index2word])

        return np.mean(data, axis=0) if len(data) else np.zeros(max_len)

    def extract_gensim_embedding(self, model: KeyedVectors.load_word2vec_format, texts: np.array,
                                 max_len: int) -> np.ndarray:
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            _array[idx] = self.extract_embedding(model, text.split(), max_len)

        return StandardScaler().fit_transform(_array)

    @property
    def embeddings(self) -> List:
        return ["gpt2", "bert", "w2v", "fastText", "glove"]

    def extract(self, name: str, values: np.array, max_len: int) -> np.ndarray:
        return {
            "gpt2": self.extract_gpt_embedding(values, self.gpt2_tokenizer, self.gpt2_model),
            "bert": self.extract_beto_embedding(self.bert_model, self.beto_tokenizer, values),
            "w2v": self.extract_gensim_embedding(self.w2b_es, values, max_len=max_len),
            "fastText": self.extract_gensim_embedding(self.ft_es, values, max_len=max_len),
            "glove": self.extract_gensim_embedding(self.glove_es, values, max_len=max_len)
        }.get(name)

    def exist_npz(self, name: str, max_len: int, x_: np.array) -> np.ndarray:
        if not os.path.exists(f"{self.numpy_path}/{name}.npz"):
            vec = self.extract(name, x_.values, max_len)
            np.savez(f"{self.numpy_path}/{name}", vec)
            self.logger.info("saved successfully", f"{self.numpy_path}/{name}.npz")
            return vec
        else:
            self.logger.info("loaded successfully")
            return np.load(f"{self.numpy_path}/{name}.npz")["arr_0"]
