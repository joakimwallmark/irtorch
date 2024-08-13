import logging
from .bit_scales import BitScales
from .evaluation import Evaluation
from .plotting import Plotting
from . import load_dataset
from . import config
from . import utils
from . import cross_validation

__version__ = "0.2.0"

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("irtorch")
