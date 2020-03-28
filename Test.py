"""
Test class. I'm too lazy to add it to the .gitignore
"""
import numpy as np

def f(x):
    return x**2

T = np.arange(10)
Y = np.array(list(map(f, T)))
print(Y)
