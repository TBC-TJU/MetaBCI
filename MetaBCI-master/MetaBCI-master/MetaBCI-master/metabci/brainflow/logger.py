# -*- coding: utf-8 -*-
# License: MIT License
"""
Logging system.

"""
import logging


def get_logger(log_name):
    """get system logger.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        Nonw
    Parameters
    ----------
    log_name: str,
        Name of logger.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler("log.txt", encoding="utf-8")
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def disable_log():
    """disable system logger.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        Nonw
    """
    logging.disable(logging.INFO)
