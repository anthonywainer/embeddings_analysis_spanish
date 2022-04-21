import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

from embeddings_analysis_spanish.evaluation.clustering_accuracy import acc
from embeddings_analysis_spanish.utils.logger import Logger


def target_distribution(q):
    """
    computing an auxiliary target distribution
    :param q: values
    :return: Adjusted
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def get_metrics(y, y_predicted, show_result=True):
    """
    Function to get Metrics
    :param y: Labels
    :param y_predicted: Labels predicted
    :param show_result: True or False
    :return: Metrics
    """
    rounded_acc = np.round(acc(y, y_predicted), 5)
    rounded_nmi = np.round(nmi(y, y_predicted), 5)
    rounded_ari = np.round(ari(y, y_predicted), 5)

    if show_result:
        Logger().logger.info(f'Metrics: ACC= {rounded_acc}, NMI= {rounded_nmi}, ARI= {rounded_ari}')
    return rounded_acc, rounded_nmi, rounded_ari
