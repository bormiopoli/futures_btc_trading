import logging
import os
import pathlib


def initialize_logger(logging, root):
    logger = logging.getLogger('FUTURES_log')
    logger.setLevel(logging.DEBUG)
    log_file_path = f'{root}/FUTURES.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w+') as log_file:
        pass
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

root = __file__.rsplit(os.sep, 1)[0]+"/file_folder"

logger = initialize_logger(logging, root)
