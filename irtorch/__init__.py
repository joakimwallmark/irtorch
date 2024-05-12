import logging
from .irt import IRT

__version__ = '0.0.5'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('irtorch')
