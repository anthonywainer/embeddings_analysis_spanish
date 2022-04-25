import numpy as np

from sklearn.preprocessing import StandardScaler
from transformers import TFGPT2Model, GPT2Tokenizer, pipeline

from embeddings_analysis_spanish.embeddings.base_embedding import BaseEmbedding


class GPTEmbedding(BaseEmbedding):
    """
    GPT Embedding
    * GPT2 - @author: datificate/gpt2-small-spanish
    """

    def __init__(self, gpt_name: str = "datificate/gpt2-small-spanish") -> None:
        """
        Init embeddings extraction
        :param numpy_path: Path to save or load vector numpy
        """

        super().__init__()
        self.gpt_name = gpt_name
        self.gpt_model = None
        self.gpt_tokenizer = None

    def extract_gpt_embedding(self, texts: np.array) -> np.ndarray:
        """
        Method to extract embeddings from GPT2
        :param texts: Text in format array numpy
        :return: dimensional array with embeddings
        """

        if self.gpt_model is None:
            self.gpt_model = TFGPT2Model.from_pretrained(self.gpt_name)
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_name, do_lower_case=True)

        max_len: int = 768
        _array = np.ndarray(shape=(len(texts), max_len), dtype=np.float32)

        for idx, text in enumerate(texts):
            self.gpt_tokenizer.pad_token = "[PAD]"
            pipe = pipeline('feature-extraction', model=self.gpt_model, tokenizer=self.gpt_tokenizer)
            features = pipe(text)
            features = np.squeeze(features)
            _array[idx] = features.mean(axis=0) if len(features) > 1 else np.zeros(max_len)

        return StandardScaler().fit_transform(_array)
