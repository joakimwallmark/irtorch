import logging
logger = logging.getLogger("irtorch")

def set_verbosity(level) -> None:
    """
    Specifies the severity of log messages that the package will display.

    Parameters
    ----------
    level : int
        The verbosity level. Possible values are:

        - 1: to only display critical messages
        - 2: to display critical messages and error messages
        - 3: to display everything from 2 and warning messages
        - 4: to display everything from 3 and info messages
        - 5: to display everything from 4 and debug messages
    """
    if level == 1:
        logger.setLevel(logging.CRITICAL)
    elif level == 2:
        logger.setLevel(logging.ERROR)
    elif level == 3:
        logger.setLevel(logging.WARNING)
    elif level == 4:
        logger.setLevel(logging.INFO)
    elif level == 5:
        logger.setLevel(logging.DEBUG)
    else:
        raise ValueError("Invalid verbosity level. Level should be 1, 2, 3, 4 or 5.")
    