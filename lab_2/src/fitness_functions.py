import numpy as np

def ackley(x):
    # ackley function
    x1, x2 = x[0], x[1]
    part1 = -0.2 * np.sqrt(0.5 * (x1**2 + x2**2))
    part2 = 0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
    return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e

def rosenbrock(x):
    # rosenbrock function
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def cross_in_tray(x):
    # cross-in-tray function
    x1, x2 = x[0], x[1]
    fact = np.sin(x1) * np.sin(x2) * np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -0.0001 * (abs(fact) + 1)**0.1

def holder_table(x):
    # holder table function
    x1, x2 = x[0], x[1]
    return -abs(np.sin(x1) * np.cos(x2) * np.exp(abs(1 - (np.sqrt(x1**2 + x2**2) / np.pi))))

def mccormick(x):
    # mccormick function
    x1, x2 = x[0], x[1]
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1

def styblinski_tang(x):
    # styblinski-tang function
    return 0.5 * sum([xi**4 - 16 * xi**2 + 5 * xi for xi in x])