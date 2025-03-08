import logging

class Logger:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def log(self, **kwargs):
        self.logger.info(kwargs)


logger = Logger()