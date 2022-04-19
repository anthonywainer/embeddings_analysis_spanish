import pandas as pd

from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

MAIN_PATH = "dataset"
dataset = pd.read_excel(f"{self.path}/processed/BBCNewsProcessed.xlsx")
max_lenh = 300  # max sequence length

le = LabelEncoder()
X_ = dataset.clean_contenido.astype(str)
Y = to_categorical(le.fit_transform(dataset.categoria))

n_clusters = len(le.classes_)
dict(enumerate(le.classes_))
