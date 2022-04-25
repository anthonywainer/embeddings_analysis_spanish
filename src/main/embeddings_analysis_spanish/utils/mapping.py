from collections.abc import Mapping
from typing import Iterator, Any


class LazyDict(Mapping):
    def __init__(self, *args, **kw) -> None:
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key: str) -> Any:
        func, arg = self._raw_dict.__getitem__(key)
        return func(*arg)

    def __iter__(self) -> Iterator:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)
