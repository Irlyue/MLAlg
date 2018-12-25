import numpy as np


def load_simple_data():
    name = 'data/testSet.txt'
    with open(name, 'rt') as f:
        items = [[float(item) for item in line.strip().split()]for line in f]
    xs = np.array([item[:2] for item in items])
    ys = np.array([item[2] for item in items])
    return xs, ys
