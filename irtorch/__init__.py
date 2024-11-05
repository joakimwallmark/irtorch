import logging
from .evaluator import Evaluator
from .plotter import Plotter
from . import load_dataset
from .config import *
from .utils import *

__version__ = "0.4.0"

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("irtorch")
