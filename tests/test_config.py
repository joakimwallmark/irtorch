import logging
from irtorch.config import set_verbosity

def test_set_verbosity():
    # Test setting verbosity level 1
    set_verbosity(1)
    assert logging.getLogger('irtorch').getEffectiveLevel() == logging.CRITICAL

    # Test setting verbosity level 2
    set_verbosity(2)
    assert logging.getLogger('irtorch').getEffectiveLevel() == logging.ERROR

    # Test setting verbosity level 3
    set_verbosity(3)
    assert logging.getLogger('irtorch').getEffectiveLevel() == logging.WARNING

    # Test setting verbosity level 4
    set_verbosity(4)
    assert logging.getLogger('irtorch').getEffectiveLevel() == logging.INFO

    # Test setting verbosity level 5
    set_verbosity(5)
    assert logging.getLogger('irtorch').getEffectiveLevel() == logging.DEBUG

    # Test setting invalid verbosity level
    try:
        set_verbosity(6)
    except ValueError as e:
        assert str(e) == "Invalid verbosity level. Level should be 1, 2, 3, 4 or 5."