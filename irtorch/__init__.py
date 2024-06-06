import logging
from .bit_scales import BitScales
from .evaluation import Evaluation
from .plotting import Plotting

__version__ = "0.1.1"

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("irtorch")
