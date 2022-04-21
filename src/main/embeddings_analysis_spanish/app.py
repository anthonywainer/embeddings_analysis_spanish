from embeddings_analysis_spanish.embedding.process_dataframes import ProcessDataFrames
from embeddings_analysis_spanish.modeling.autoencoder_model import AutoencoderModel
from embeddings_analysis_spanish.modeling.k_means_model import KMeansModel

path = ""
process = ProcessDataFrames(path)
dataset_embeddings = process.start_process()

kmeans_model = KMeansModel(path)
kmeans_model.run(dataset_embeddings)

autoencoder_model = AutoencoderModel(path)
autoencoder_model.run(dataset_embeddings)
