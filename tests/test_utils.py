import pytest
from irtorch.utils import dynamic_print, is_jupyter

def test_is_jupyter():
    assert is_jupyter() is False

def test_dynamic_print(capsys):
    dynamic_print("arg1\narg2")

    # Capture the output again after calling dynamic_print
    captured_output = capsys.readouterr()
    assert captured_output.out == '\rarg1\narg2 '

    dynamic_print("Hello World!")
    captured_output = capsys.readouterr()

    assert captured_output.out == "\rHello World! "
