from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class MetricsModel:
    acc: float
    nmi: float
    ari: float

    @property
    def to_tuple(self) -> Tuple:
        return tuple([self.__dict__[field.name] for field in fields(self)])
