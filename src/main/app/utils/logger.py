import logging


class Logger(object):
    """
    Wrapper class for logger
    """
    __instance = None
    logger = None

    def __int__(self) -> None:
        self.logger = logging.getLogger("embeddings")
        log_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=log_format)
        self.logger.setLevel(logging.INFO)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error.

        :param message: Error message to write to log
        """

        self.logger.error(message, *args, **kwargs)

    def warn(self, message: str, *args, **kwargs) -> None:
        """Log warning.

        :param message: Error message to write to log
        """

        self.logger.warning(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log information.

        :param message: Information message to write to log
        """

        self.logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log information.

        :param message: Debug message to write to log
        """

        self.logger.debug(message, *args, **kwargs)
