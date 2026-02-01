import unittest
import numpy as np
from src.calc import add, subtract, multiply, divide, square_root


class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(add(2, 3), 5)

    def test_fun2(self):
        self.assertEqual(subtract(10, 4), 6)

    def test_fun3(self):
        self.assertEqual(multiply(3, 5), 15)

    def test_fun4(self):
        self.assertEqual(divide(10, 2), 5)

    def test_square_root_negative(self):
        with self.assertRaises(ValueError):
            square_root(-9)


if __name__ == "__main__":
    unittest.main()
