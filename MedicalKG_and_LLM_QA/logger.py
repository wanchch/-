import os
import logging
from pathlib import Path

log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_level_value = os.environ.get("LOG_LEVEL", logging.INFO)


logging.basicConfig(level=logging.INFO,
                 format=log_format,
                 datefmt='%Y-%m-%d %H:%M:%S',
                filename='log.txt',
                filemode='a')

