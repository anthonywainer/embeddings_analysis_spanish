import os
from typing import List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler

from transformers import TFGPT2Model, GPT2Tokenizer, BertTokenizer, TFBertModel
from transformers import pipeline
from gensim.models import KeyedVectors

MAIN_PATH = "dataset"
dataset = pd.read_excel(f"{MAIN_PATH}/processed/BBCNewsProcessed.xlsx")
max_len = 300  # max sequence length

le = LabelEncoder()
X_ = dataset.clean_contenido.astype(str)
Y = to_categorical(le.fit_transform(dataset.categoria))

n_clusters = len(le.classes_)
dict(enumerate(le.classes_))


class BaseEmbedding(object):

    def __init__(self) -> None:
        gpt2 = "datificate/gpt2-small-spanish"

        self.gpt2_model = TFGPT2Model.from_pretrained(gpt2)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2, do_lower_case=True)

        bert = "dccuchile/bert-base-spanish-wwm-uncased"
        self.bert_model = TFBertModel.from_pretrained(bert)
        self.beto_tokenizer = BertTokenizer.from_pretrained(bert, do_lower_case=True)

        self.w2b_es = KeyedVectors.load_word2vec_format(f'{MAIN_PATH}/SBW-vectors-300-min5.txt')
        self.ft_es = KeyedVectors.load_word2vec_format(f'{MAIN_PATH}/embeddings-l-model.vec')
        self.glove_es = KeyedVectors.load_word2vec_format(f'{MAIN_PATH}/glove-sbwc.i25.vec')

    @staticmethod
    def extract_gpt_embedding(texts: np.array, tokenizer, model, max_len: int = 768):
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            tokenizer.pad_token = "[PAD]"
            pipe = pipeline('feature-extraction', model=model,
                            tokenizer=tokenizer)
            features = pipe(text)
            features = np.squeeze(features)
            _array[idx] = features.mean(axis=0) if len(features) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_beto_embedding(model, tokenizer, texts: np.array, max_len: int = 768):
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            input_ids = tf.constant(tokenizer.encode(text))[None, :]
            outputs = model(input_ids)
            last_hidden_states = outputs["pooler_output"]
            cls_token = last_hidden_states[0]

            _array[idx] = np.mean(cls_token, axis=0) if len(cls_token) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)

    @staticmethod
    def extract_embedding(model, words, max_len):
        data = np.array([model.wv[w] for w in words if w in model.wv.index2word])

        return np.mean(data, axis=0) if len(data) else np.zeros(max_len)

    def extract_gensim_embedding(self, model, texts: np.array, max_len):
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            _array[idx] = self.extract_embedding(model, text.split(), max_len)

        return StandardScaler().fit_transform(_array)

    @property
    def embeddings(self) -> List:
        return ["gpt2", "bert", "w2v", "fastText", "glove"]

    def extract(self, name, values):
        return {
            "gpt2": self.extract_gpt_embedding(values, self.gpt2_tokenizer, self.gpt2_model),
            "bert": self.extract_beto_embedding(self.bert_model, self.beto_tokenizer, values),
            "w2v": self.extract_gensim_embedding(self.w2b_es, values, max_len=max_len),
            "fastText": self.extract_gensim_embedding(self.ft_es, values, max_len=max_len),
            "glove": self.extract_gensim_embedding(self.glove_es, values, max_len=max_len)
        }.get(name)

    def exist_npz(self, path: str, name: str):
        if not os.path.exists(f"numpy_data/{path}/{name}.npz"):
            vec = self.extract(name, X_.values)
            np.savez(f"numpy_data/{path}/{name}", vec)
            print("saved successfully")
            return vec
        else:
            print("loaded successfully")
            return np.load(f"numpy_data/{path}/{name}.npz")["arr_0"]
