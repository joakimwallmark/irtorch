import logging
logger = logging.getLogger('irtorch')

def set_verbosity(level):
    """
    Set the verbosity level for the package's logging.
    
    Parameters
    ----------
    level : int
        The verbosity level. 0 is warning, 1 is info, 2 is debug.
    """
    if level == 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    elif level >= 2:
        logger.setLevel(logging.DEBUG)
    else:
        raise ValueError("Invalid verbosity level. Level should be 0, 1, or 2.")
    
# TODO pandas mode?
