from embeddings_analysis_spanish.cleaning.base_cleaning import BaseCleaning
from embeddings_analysis_spanish.utils.cleaner import processing_words


class BBCCleaning(BaseCleaning):
    """
    Cooking bbc_news dataset
    """

    def process(self) -> None:
        dataset = self.read_dataframe(f"{self.path}/translated/bbc_news_es.xlsx")

        dataset.loc[:, 'clean_news'] = dataset.content.apply(processing_words)
        dataset["id"] = range(0, len(dataset))

        dataset = dataset.drop_duplicates(subset='clean_news', keep="last")
        dataset = dataset.sort_values(by=["category"])

        self.write_dataframe(dataset, f"{self.path}/processed/bbc_news_processed.xlsx")
