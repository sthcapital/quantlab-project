import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("quantlab")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base_logger = logging.getLogger("quantlab")
    if name:
        return base_logger.getChild(name)
    return base_logger