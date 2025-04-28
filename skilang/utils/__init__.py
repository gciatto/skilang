import logging
from typing import Set

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("skilang")

DEFAULT_LOG_LEVEL = logging.DEBUG


def _all_loggers(root=logging.root) -> Set[logging.Logger]:
    return {root} | {logging.getLogger(logger) for logger in root.manager.loggerDict}


def _our_loggers():
    result = set()
    logger_tmp = logger
    while logger_tmp is not None:
        result.add(logger_tmp)
        logger_tmp = logger_tmp.parent
    return result


def set_loggers_level(selector, level: int | str = DEFAULT_LOG_LEVEL):
    if isinstance(level, str):
        level = logging.ERROR
    for logger in _all_loggers():
        if selector(logger):
            logger.setLevel(level)


def set_other_loggers_level(level: int | str = logging.ERROR):
    set_loggers_level(lambda logger: logger not in _our_loggers(), level)


set_other_loggers_level(logging.WARNING)
