import numpy as np
import pytest
from src.calc import add, subtract, multiply, divide, square_root


def test_fun1():
    assert add(2, 3) == 5


def test_fun2():
    assert subtract(10, 4) == 6


def test_fun3():
    assert multiply(3, 5) == 15


def test_fun4():
    assert divide(10, 2) == 5

def test_square_root_negative():
    with pytest.raises(ValueError):
        square_root(-4)
