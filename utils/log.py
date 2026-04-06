# utils/log.py
"""
Application-wide logger configuration.

Call setup_logger() once at application startup; all 
modules then use logging.getLogger(__name__) as usual.

Usage:
    from utils.log import setup_logger
    setup_logger(cfg)

    # in any module:
    import logging
    log = logging.getLogger(__name__)
    log.info("pipeline started")
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from config import Config

_LOG_FORMAT = "%(asctime)s (%(levelname)-8s) [%(filename)-14s:%(lineno)-3d] - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# top-level packages that belong to this project are listed. all others are 
# treated as external libraries and will only emit WARNING and above messages
_PROJECT_PACKAGES = ("config", "src", "utils", "__main__")

# tracks whether setup has already been called
_initialised = False

def setup_logger(config: Config, level: int = logging.DEBUG) -> logging.Logger:
    """
    configure the root logger with file and stream handlers.

    the log file is created in config.system["log_dir"] with a name based
    on the current timestamp. the directory is created if it does not exist.

    calling this function more than once is safe; subsequent calls are
    ignored so handlers are not duplicated.
    """
    global _initialised
    if _initialised:
        return logging.getLogger()

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # stream handler (stderr)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # file handler (timestamped log file)
    log_dir = Path(config.system["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sorter_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for pkg in _PROJECT_PACKAGES:
        logging.getLogger(pkg).setLevel(level)

    _initialised = True

    logging.getLogger(__name__).info("logger initialised, writing to %s", log_file)
    return root
