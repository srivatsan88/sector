import logging
import sys

# Set up the logger
logger = logging.getLogger("sector_logger")
logger.setLevel(logging.INFO)  # Default level

# Configure the handler to output to stdout
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def set_log_level(level_name):
    """Sets the logging level for both the logger and handler."""
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level_name}")
    
    logger.setLevel(level)
    stream_handler.setLevel(level)
    logger.info(f"Log level set to {level_name.upper()}")

# Expose logger as a module-level variable
__all__ = ["logger", "set_log_level"]

