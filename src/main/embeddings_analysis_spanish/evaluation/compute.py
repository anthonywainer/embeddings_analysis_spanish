import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

from embeddings_analysis_spanish.evaluation.clustering_accuracy import acc
from embeddings_analysis_spanish.models.metrics_model import MetricsModel
from embeddings_analysis_spanish.utils.logger import Logger


def target_distribution(values: np.ndarray) -> np.ndarray:
    """
    computing an auxiliary target distribution
    :param values: values
    :return: Adjusted values
    """
    weight = values ** 2 / values.sum(0)
    return (weight.T / weight.sum(1)).T


def get_metrics(y_true: np.ndarray, y_predicted: np.ndarray, show_result=True) -> MetricsModel:
    """
    Function to get Metrics
    :param y_true: Labels
    :param y_predicted: Labels predicted
    :param show_result: True or False
    :return: Metrics
    """
    rounded_acc = np.round(acc(y_true, y_predicted), 5)
    rounded_nmi = np.round(nmi(y_true, y_predicted), 5)
    rounded_ari = np.round(ari(y_true, y_predicted), 5)

    if show_result:
        Logger().logger.info(f'Metrics: ACC= {rounded_acc}, NMI= {rounded_nmi}, ARI= {rounded_ari}')
    return MetricsModel(rounded_acc, rounded_nmi, rounded_ari)
