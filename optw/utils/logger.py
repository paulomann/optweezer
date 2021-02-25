import termcolor
import logging
import copy
import os
from optw import settings


def set_logger(context, verbose=False):
    """Return colored logger with specified context name and debug=verbose"""
    # We do not use windows, ew.
    # if os.name == 'nt':  # for Windows
    #     return NTLogger(context, verbose)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(settings.LOG_PATH)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s:%(levelname)-.1s:' + context +
            ':[%(filename).5s:%(funcName).5s:%(lineno)3d]:%(message)s',
            datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger = logging.getLogger(context)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = ColoredFormatter(
            '%(asctime)s>%(levelname)-.1s:' + context +
            ':[%(filename).5s:%(funcName).5s:%(lineno)3d]:%(message)s',
            datefmt='%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.handlers = []
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


class ColoredFormatter(logging.Formatter):
    """Format log levels with color"""
    MAPPING = {
        'DEBUG': dict(color='green', on_color=None),
        'INFO': dict(color='cyan', on_color=None),
        'WARNING': dict(color='yellow', on_color=None),
        'ERROR': dict(color='grey', on_color='on_red'),
        'CRITICAL': dict(color='grey', on_color='on_blue'),
    }

    PREFIX = '\033['
    SUFFIX = '\033[0m'

    def format(self, record):
        """Add log ansi colors"""
        crecord = copy.copy(record)
        seq = self.MAPPING.get(crecord.levelname, self.MAPPING['INFO'])
        crecord.msg = termcolor.colored(crecord.msg, **seq)
        return super().format(crecord)