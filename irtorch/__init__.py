import logging

__version__ = "0.1.0"

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("irtorch")
