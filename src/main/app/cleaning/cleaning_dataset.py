from .bbc_cleaning import BBCCleaning
from .complaints_cleaning import ComplaintsCleaning
from .food_cleaning import FoodCleaning
from .imdb_cleaning import IMDBCleaning
from .scopus_cleaning import ScopusCleaning

if __name__ == "__main__":
    BBCCleaning().process()
    ComplaintsCleaning().process()
    IMDBCleaning().process()
    ScopusCleaning().process()
    FoodCleaning().process()
