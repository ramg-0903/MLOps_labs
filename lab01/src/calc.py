import numpy as np

def add(a, b):
    return np.add(a, b)

def subtract(a, b):
    return np.subtract(a, b)

def multiply(a, b):
    return np.multiply(a, b)

def divide(a, b):
    if np.any(b == 0):
        raise ValueError("Division by zero is not allowed")
    return np.divide(a, b)


def square_root(a):
    if np.any(a < 0):
        raise ValueError("Square root of negative number is not allowed")
    return np.sqrt(a)


