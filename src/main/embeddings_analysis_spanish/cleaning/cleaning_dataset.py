from embeddings_analysis_spanish.cleaning.bbc_cleaning import BBCCleaning
from embeddings_analysis_spanish.cleaning.complaints_cleaning import ComplaintsCleaning
from embeddings_analysis_spanish.cleaning.food_cleaning import FoodCleaning
from embeddings_analysis_spanish.cleaning.imdb_cleaning import IMDBCleaning
from embeddings_analysis_spanish.cleaning.scopus_cleaning import ScopusCleaning

if __name__ == "__main__":
    BBCCleaning().process()
    ComplaintsCleaning().process()
    IMDBCleaning().process()
    ScopusCleaning().process()
    FoodCleaning().process()
