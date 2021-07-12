import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import tensorflow as tf
import warnings

from time import time

from IPython.core.display import display
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, \
    roc_auc_score, classification_report, normalized_mutual_info_score, \
    adjusted_rand_score

from tensorflow import int32
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Layer, InputSpec
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
    ModelCheckpoint
from tensorflow.keras import backend as K

from mlxtend.plotting import plot_confusion_matrix

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.cleaner import processing_words
from transformers import BertTokenizer, TFBertModel


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def scaler_np(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def norma(data):
    t = Normalizer().fit(data)
    return t.transform(data)


def autoencoder(dims, input, inputs, act=tf.nn.leaky_relu, init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    h = input

    for i in range(n_stacks - 1):
        h = tf.keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    h = tf.keras.layers.Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    y = h
    for i in range(n_stacks - 1, 0, -1):
        y = tf.keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    y = tf.keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return tf.keras.models.Model(inputs=inputs, outputs=y, name='AE'), tf.keras.models.Model(inputs=inputs, outputs=h,
                                                                                             name='encoder')


path = "data/stackoverflow"

dataset = pd.read_excel(f"{path}/stackoverflow_dataset_es.xlsx")

display(dataset.sort_index().head(5))
print('\n Hay {} observaciones con {} características'.format(*dataset.shape))

# Limpieza
dataset.loc[:, ('clean_text')] = dataset.text.apply(processing_words)
dataset = dataset.drop_duplicates(subset='clean_text', keep="last")
dataset.sort_index(inplace=True)

display(dataset.sort_index().head(5))
print('\n Hay {} observaciones con {} características'.format(*dataset.shape))

# BERT
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
max_len = 256  # max sequence length
le = LabelEncoder()

X_ = dataset.clean_text
Y_ = to_categorical(le.fit_transform(dataset.label))

print(dict(enumerate(le.classes_)))

# To BERT vector
X = bert_encode(X_.values, tokenizer, max_len=max_len)
Y = Y_
X_ = norma(X[0]), X[1], X[2]
print(X_[0][0])
# BERT layer
input_word_ids = Input(shape=(max_len,), dtype=int32, name="input_word_ids")
input_mask = Input(shape=(max_len,), dtype=int32, name="input_mask")
segment_ids = Input(shape=(max_len,), dtype=int32, name="segment_ids")
bert_inputs = [input_word_ids, input_mask, segment_ids]
bert_output = TFBertModel.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-uncased", trainable=True,
    name=f"beto_model"
)(bert_inputs)["pooler_output"]

dims = [X[0].shape[-1], 500, 500, 2000, 20]
autoencoder_model, encoder_model = autoencoder(dims, bert_output, bert_inputs)

pretrain_optimizer = Adam(learning_rate=1e-5)
autoencoder_model.compile(optimizer=pretrain_optimizer, loss='mse', metrics="mse")
autoencoder_model.summary()

t0 = time()
#tf.random.set_seed(73)
#np.random.seed(73)

print(X_[0].shape, X_[1].shape, X_[2].shape)
history = autoencoder_model.fit(
    X_, X_,
    epochs=5,
    batch_size=1
)
autoencoder_model.save_weights(f'{path}/ae/ae_weights.h5', overwrite=True)
print('Pretraining time: %ds' % round(time() - t0))
