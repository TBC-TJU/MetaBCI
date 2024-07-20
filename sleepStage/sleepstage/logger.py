import os
import logging as py_logging


_log_level = {
    None: py_logging.NOTSET,
    "debug": py_logging.DEBUG,
    "info": py_logging.INFO,
    "warning": py_logging.WARNING,
    "error": py_logging.ERROR,
    "critical": py_logging.CRITICAL
}


def get_logger(
    log_file_path=None,
    name="default_log",
    level=None
):
    directory = os.path.dirname(log_file_path)
    if os.path.isdir(directory) and not os.path.exists(directory):
        os.makedirs(directory)

    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = py_logging.Formatter(f"{level_format} %(message)s")
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger


if __name__ == "__main__":
    logger = get_logger("test.log", name="test", level="info")
    logger.info("Test")
    logger.info("Test2")
    logger.info("Test3")
