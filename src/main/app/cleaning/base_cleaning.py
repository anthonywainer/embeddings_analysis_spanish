from typing import Any


class BaseCleaning(object):

    def __int__(self, path: str = "data/dataset") -> None:
        self.path = path

    @staticmethod
    def _count_words(text: Any) -> int:
        """
        Count words very easy, valid first is None and if is string
        @param text: Text to process
        @return: 0 if is None
        """
        if (text is not None) and (isinstance(text, str)):
            words = set(text.split())
            if words:
                return len(words)
        return 0
