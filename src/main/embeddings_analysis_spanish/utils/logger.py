import logging


class Logger(object):
    """
    Wrapper class for logger
    """
    logger = None

    def __init__(self) -> None:
        self.logger = logging.getLogger("embeddings")
        log_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=log_format)
        self.logger.setLevel(logging.INFO)
