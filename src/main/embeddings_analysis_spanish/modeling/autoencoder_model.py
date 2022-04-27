from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from keras import Model
from keras.activations import leaky_relu
from keras.layers import Dense, Input
from keras.optimizer_v1 import SGD
from keras.optimizer_v2.adam import Adam
from sklearn.cluster import KMeans

from embeddings_analysis_spanish.evaluation.compute import target_distribution, get_metrics
from embeddings_analysis_spanish.modeling.base_model import BaseModel
from embeddings_analysis_spanish.modeling.clustering_layer import ClusteringLayer
from embeddings_analysis_spanish.utils.mapping import LazyDict


class AutoencoderModel(BaseModel):
    """
    Autoencoder Model
        Fit and predict on batch
    """

    def __init__(self, path: str = "data/results",
                 max_iteration: int = 100, update_interval: int = 140,
                 tolerance_threshold: float = 1e-3, batch_size: int = 512) -> None:
        """
        Initializing vars
        :param max_iteration: Max Interation
        :param update_interval: Update Interval
        :param tolerance_threshold: Tolerance threshold to stop training
        :param batch_size: Batch Size
        """

        super().__init__(path)
        self.model_name = "autoencoder_kmeans"
        self.max_iteration = max_iteration
        self.update_interval = update_interval
        self.tolerance_threshold = tolerance_threshold
        self.batch_size = batch_size

        tf.random.set_seed(73)
        np.random.seed(73)

    @staticmethod
    def __build_encoder(input_model: Input, n_stacks: int,
                        dimensions: List, activation: str = leaky_relu) -> Dense:
        """
        Building encoder
        :param input_model: Input Model TF
        :param n_stacks: Position
        :param dimensions: weight layer
        :param activation: Activation Function
        :return: Model TF
        """

        for i in range(n_stacks - 1):
            input_model = Dense(
                dimensions[i + 1],
                activation=activation,
                name='encoder_%d' % i
            )(input_model)

        return Dense(
            dimensions[-1],
            name='encoder_%d' % (n_stacks - 1)
        )(input_model)

    @staticmethod
    def __build_decoder(decoder_input: Dense, n_stacks: int, dimensions: List,
                        activation: str = leaky_relu) -> Dense:
        """
        Building encoder
        :param decoder_input: Input Model TF
        :param n_stacks: Position
        :param dimensions: weight layer
        :param activation: Activation Function
        :return: Model TF
        """

        for i in range(n_stacks - 1, 0, -1):
            decoder_input = Dense(
                dimensions[i],
                activation=activation,
                name='decoder_%d' % i
            )(decoder_input)

        return Dense(
            dimensions[0],
            name='decoder_0'
        )(decoder_input)

    def __autoencoder_layer(self, dimensions: List, input_model: Input,
                            activation: str = leaky_relu,
                            name: str = "") -> Tuple[Model, Model]:
        """
        Layer AutoEncoder
        :param dimensions:  weight layer
        :param input_model: Input Model TF
        :param activation:  Activation Function
        :param name: Model Name
        :return: Autoencoder Layer, Encoder Layer
        """

        n_stacks = len(dimensions) - 1
        encoder = self.__build_encoder(input_model, n_stacks, dimensions, activation)
        decoder = self.__build_decoder(encoder, n_stacks, dimensions, activation)

        return (
            Model(inputs=input_model, outputs=decoder, name=f'AE-{name}'),
            Model(inputs=input_model, outputs=encoder, name=f'encoder-{name}')
        )

    def fit_autoencoder_model(self, dimensions: List, embedding: np.ndarray, name: str,
                              verbose: bool = False, epochs: int = 200, batch_size: int = 512):
        """
        Fit Autoencoder Model
        :param batch_size: The batch size
        :param dimensions:  weight layer
        :param embedding: Embeddings extracted from datasets
        :param name: Model Name
        :param verbose: show summary model
        :param epochs: Epochs to fit
        :return: autoencoder_model, encoder_model
        """

        input_model = Input(shape=(dimensions[0],), name='input')
        autoencoder_model, encoder_model = self.__autoencoder_layer(dimensions, input_model, name=name)
        pretrain_optimizer = Adam(learning_rate=1e-4)
        autoencoder_model.compile(optimizer=pretrain_optimizer, loss='mse', metrics="mse")

        if verbose:
            autoencoder_model.summary()

        autoencoder_model.fit(
            embedding, embedding,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        return autoencoder_model, encoder_model

    def predict_on_batch(self, embedding: np.ndarray, model: Model,
                         y_predicted_last: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
        """
        Predict On Batch
        :param embedding: Embeddings extracted from datasets
        :param model: Model
        :param y_true: Labels
        :param y_predicted_last: Result
        :return: Result predicted
        """

        loss = 0
        index = 0
        adjusted = None
        y_predicted = None
        index_array = np.arange(embedding.shape[0])

        for iterator in range(int(self.max_iteration)):
            if iterator % self.update_interval == 0:
                predicted = model.predict(embedding, verbose=1)
                adjusted = target_distribution(predicted)

                y_predicted = predicted.argmax(1)
                if y_true:
                    y_true_adjusted = np.argmax(y_true, axis=1)

                    get_metrics(y_true_adjusted, y_predicted)
                    self.logger.info(f'loss = {loss}')

                delta_label = np.sum(y_predicted != y_predicted_last).astype(np.float32) / y_predicted.shape[0]
                y_predicted_last = np.copy(y_predicted)

                if iterator > 0 and delta_label < self.tolerance_threshold:
                    self.logger.info('delta_label ', delta_label, '< tol ', self.tolerance_threshold)
                    self.logger.info('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size, embedding.shape[0])]
            loss = model.train_on_batch(x=embedding[idx], y=adjusted[idx])
            index = index + 1 if (index + 1) * self.batch_size <= embedding.shape[0] else 0

        return y_predicted

    @property
    def dimensions(self) -> List:
        return [None, 500, 2000, 5000, 10000, 100]

    def predict_encoder(self, dimensions: List, embedding: np.ndarray, name: str, epochs: int = 500) -> Tuple:
        autoencoder_model, encoder_model = self.fit_autoencoder_model(
            dimensions,
            embedding,
            name,
            epochs=epochs
        )
        return encoder_model.predict(embedding, verbose=1), encoder_model

    def __compile_clustering_layer(self, encoder_model: Model, cluster_number: int, name: str) -> Model:
        self.logger.info(f'Clustering - Layer Custom {name}')
        clustering_layer = ClusteringLayer(
            cluster_number, name=f'clustering-{name}'
        )(encoder_model.output)

        model = Model(inputs=encoder_model.input, outputs=clustering_layer)
        model.compile(SGD(0.1, 0.9), loss='kld')

        return model

    def __fit_predict_kmeans(self, predict_encoder: Model, cluster_number: int) -> Tuple:
        self.logger.info('Clustering - K-means')
        kmeans = KMeans(n_clusters=cluster_number, max_iter=100, random_state=73)
        y_predicted = kmeans.fit_predict(predict_encoder)

        return kmeans, y_predicted

    def fit_predict(self, embedding: np.ndarray, name: str,
                    cluster_number: int, y_true: np.ndarray,
                    predicted_embedding: Dict, data_metrics: List):

        dimensions = self.dimensions
        dimensions[0] = embedding.shape[-1]

        predict_encoder, encoder_model = self.predict_encoder(
            dimensions,
            embedding,
            f"{name}"
        )
        model = self.__compile_clustering_layer(encoder_model, cluster_number, f"{name}")
        kmeans, y_predicted = self.__fit_predict_kmeans(predict_encoder, cluster_number)

        model.get_layer(name=f'clustering-{name}').set_weights([kmeans.cluster_centers_])

        predicted_embedding[name] = (embedding, y_predicted, model)

        data_metrics.append(
            list((name,) + get_metrics(y_true.argmax(1), y_predicted).to_tuple)
        )

        return data_metrics, predicted_embedding

    def run(self, dataset_embeddings: LazyDict, save_results: bool = False) -> Tuple:
        return super().run(dataset_embeddings, save_results)
