from .BBCCleaning import BBCCleaning
from .ComplaintsCleaning import ComplaintsCleaning
from .FoodCleaning import FoodCleaning
from .IMDBCleaning import IMDBCleaning
from .ScopusCleaning import ScopusCleaning

if __name__ == "__main__":
    BBCCleaning().process()
    ComplaintsCleaning().process()
    IMDBCleaning().process()
    ScopusCleaning().process()
    FoodCleaning().process()
