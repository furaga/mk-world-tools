import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_FILE = ".cache/log.txt"


def setup_logger(
    name: str, log_file: str = DEFAULT_LOG_FILE, level: int = logging.INFO
):
    """
    Sets up a logger that outputs to both console and a file.
    """
    # Create .cache directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    # Create console handler
    stream_hander = logging.StreamHandler()
    stream_hander.setLevel(level)
    stream_formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    )
    stream_hander.setFormatter(stream_formatter)
    logger.addHandler(stream_hander)

    # Create file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )  # 10MB per file, 5 backup files
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Example usage:
    logger = setup_logger(__name__)
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    custom_logger = setup_logger(
        name="custom", log_file=".cache/custom_log.txt", level=logging.DEBUG
    )
    custom_logger.debug("This is a debug message for the custom logger.")
    custom_logger.info("This is an info message for the custom logger.")
