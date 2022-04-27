from functools import wraps
from typing import Callable

from pathlib import Path


def is_path(func: Callable) -> Callable:
    @wraps(func)
    def func_wrapper(*args, **kwargs) -> Path:
        path = Path(func(*args, **kwargs))
        if not path.is_file():
            raise FileNotFoundError(f"Path invalid! {path}")

        return path

    return func_wrapper
