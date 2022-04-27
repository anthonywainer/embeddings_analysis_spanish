from typing import Tuple, Dict

import numpy as np
from keras import backend
from keras.layers import Layer, InputSpec


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))

        # Make sure each sample's 10 values add up to 1.
    ```
    @author: Hadifar - https://github.com/hadifar/stc_clustering/blob/master/STC.py
    """

    def __init__(self, n_clusters: int, weights: np.array = None, alpha: float = 1.0, **kwargs) -> None:
        """
        Initializing cluster
        :param n_clusters: The clusters number.
        :param weights: List of Numpy array with shape `(n_clusters, n_features)`
                        witch represents the initial cluster centers.
        :param alpha: Degrees of freedom parameter in Student's t-distribution. Default to 1.0.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.built = None

    def build(self, input_shape: Tuple) -> None:
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=backend.floatx(), shape=(None, input_dim))
        self.n_clusters = self.add_weight(
            shape=(self.n_clusters, input_dim,),
            initializer='glorot_uniform',
            name=self.name
        )

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs: np.ndarray, **kwargs) -> backend.transpose:
        """
        Student T-distribution, as same as used in t-SNE algorithm.
        Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        :param inputs: the variable containing data, shape=(n_samples, n_features)
        :return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (
                backend.sum(
                    backend.square(backend.expand_dims(inputs, axis=1) - self.n_clusters), axis=2
                ) / self.alpha
        ))

        q **= (self.alpha + 1.0) / 2.0

        return backend.transpose(backend.transpose(q) / backend.sum(q, axis=1))

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self) -> Dict:
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(base_config.items() + config.items())
