import logging
from pathlib import Path


def setup_logging() -> None:
    """
    Set up dedicated loggers for database and application operations.
    Ensures clean handler setup to prevent duplicate logs.
    """

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    _setup_logger(
        name="db_logger",
        log_file=log_dir / "db_operations.log",
    )
    _setup_logger(
        name="app_logger",
        log_file=log_dir / "app_operations.log",
    )


def _setup_logger(name: str, log_file: Path) -> None:
    """Help to set up an individual logger."""
    logger = logging.getLogger(name)

    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    logger.propagate = False

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
