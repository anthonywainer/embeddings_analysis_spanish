from dataclasses import dataclass


@dataclass
class DataframeModel:
    dataframe_name: str
    clean_field: str
    category_field: str
