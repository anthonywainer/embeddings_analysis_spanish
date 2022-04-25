from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from embeddings_analysis_spanish.evaluation.compute import get_metrics


def plot_confusion_matrix(confusion_matrix: np.ndarray, name: str, save=False, path: str = "") -> None:
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 3})
    plt.title(f"Confusion matrix {name}", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()
    if save:
        plt.savefig(f"{path}/confusion_matrix_{name}.png")


def rearrange_labels(values: np.ndarray, cluster_labels: np.ndarray,
                     labels_encoders: Dict,
                     sort_column: int = 0) -> Tuple:
    """
        Rearrange Label for clustering plotting
    :param values: The values to cluster
    :param cluster_labels: Cluster Labels
    :param labels_encoders: Labels Encoders
    :param sort_column: Sort Column
    :return: New values to plot
    """

    labels, ctrs = [], []
    for i in range(len(set(cluster_labels))):
        ctr = np.mean(values[cluster_labels == i], axis=0)
        labels.append(i)
        ctrs.append(ctr)

    ctrs = np.row_stack(ctrs)
    labels = np.array(labels).reshape(-1, 1)

    new_order = ctrs[:, sort_column].argsort()
    labels_new = labels[new_order]
    ctrs_new = ctrs[new_order]

    np.put(cluster_labels, labels, labels_new)
    return cluster_labels, ctrs_new, [labels_encoders.get(n).title() for n in cluster_labels]


def plot_clustering_with_labels(embedding: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray,
                                name: str, label_encoders: Dict, save=False, path: str = "") -> None:
    """
    Plotting clustering between y_true and y_predicted
    :param embedding: The embeddings values
    :param y_true: Original values
    :param y_predicted: Values predicted
    :param name: The name embeddings to show in plot
    :param label_encoders: label in dict format with position key and label how value ie. {1: label1, 2: label2, ....}
    :param save: If define in True must configure path in the next param
    :param path: for save plot
    """

    embedding = TSNE().fit_transform(embedding)
    cluster_y_true, y_true_ctrs, y_true_labels = rearrange_labels(
        values=embedding, cluster_labels=y_true.argmax(1), labels_encoders=label_encoders
    )
    cluster_y_predicted, y_predicted_ctrs, y_predicted_labels = rearrange_labels(
        values=embedding, cluster_labels=y_predicted, labels_encoders=label_encoders
    )

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    left_scatter = ax[0].scatter(embedding[:, 0], embedding[:, 1], c=cluster_y_true)
    right_scatter = ax[1].scatter(embedding[:, 0], embedding[:, 1], c=cluster_y_predicted, cmap=plt.cm.Set1)

    metrics = get_metrics(y_true.argmax(1), y_predicted, 0)
    ax[0].set_title(f'{name} Process Analysis', fontsize=18)
    ax[1].set_title(
        "AE + Kmeans - ACC={0:.2f}, NMI={1:.2f}, ARI={2:.2f}".format(*metrics.to_tuple),
        fontsize=12
    )

    ax[0].legend(handles=left_scatter.legend_elements()[0], labels=y_true_labels)
    ax[1].legend(handles=right_scatter.legend_elements()[0], labels=y_predicted_labels)

    plt.show()
    if save:
        plt.savefig(f"{path}/clustering_plot_{name}.png")
