from embeddings_analysis_spanish.cleaning.process_dataframes import ProcessDataFrames
from embeddings_analysis_spanish.embeddings.extracting_embedding import ExtractingEmbedding
from embeddings_analysis_spanish.modeling.autoencoder_model import AutoencoderModel
from embeddings_analysis_spanish.modeling.hdbscan_model import HDBSCANModel
from embeddings_analysis_spanish.modeling.k_means_model import KMeansModel


def main() -> None:
    process_dataframes = ProcessDataFrames()

    extracting_embedding = ExtractingEmbedding()
    dataset_embeddings = extracting_embedding.start_process(
        process_dataframes.dataframes, process_dataframes.dataframes_params
    )

    kmeans_model = KMeansModel()
    kmeans_model.run(dataset_embeddings)

    autoencoder_model = AutoencoderModel()
    autoencoder_model.run(dataset_embeddings)

    hdbscan_model = HDBSCANModel()
    hdbscan_model.run(dataset_embeddings)

    if __name__ == "__main__":
        main()
