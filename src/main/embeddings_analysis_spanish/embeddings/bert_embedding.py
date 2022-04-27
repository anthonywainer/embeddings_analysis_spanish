from typing import Tuple

import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import StandardScaler

from embeddings_analysis_spanish.embeddings.base_embedding import BaseEmbedding


class BertEmbedding(BaseEmbedding):
    """
    Bert Embedding
    * BETO - @author: dccuchile/bert-base-spanish-wwm-uncased
    """

    def __init__(self, bert_name: str = "dccuchile/bert-base-spanish-wwm-uncased") -> None:
        """
        Init embeddings extraction
        """

        super().__init__()
        self.bert_name = bert_name
        self.bert_model = None
        self.bert_tokenizer = None

    def load_pretrained(self) -> Tuple[TFBertModel.from_pretrained, BertTokenizer.from_pretrained]:
        self.bert_model = TFBertModel.from_pretrained(self.bert_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_name, do_lower_case=True)

        return self.bert_model, self.bert_tokenizer

    def extract_bert_embedding(self, texts: np.array) -> np.ndarray:
        """
        Method to extract embeddings from Bert
        :param texts: Text in format array numpy
        :return: dimensional array with embeddings
        """
        if self.bert_model is None:
            self.bert_model, self.bert_tokenizer = self.load_pretrained()

        max_len = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            input_ids = tf.constant(self.bert_tokenizer.encode(text))[None, :]
            outputs = self.bert_model(input_ids)
            last_hidden_states = outputs["pooler_output"]
            cls_token = last_hidden_states[0]

            _array[idx] = np.mean(cls_token, axis=0) if len(cls_token) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)
