import numpy as np

def branin(x, y):
    # Branin-Hoo
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (y - b * x**2 + c * x - r)**2 + s * (1 - t) * np.cos(x) + s

def easom(x, y):
    # Easom
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def goldstein_price(x, y):
    # Goldstein-Price
    c1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x*x - 14*y + 6*x*y + 3*y*y)
    c2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x*x + 48*y - 36*x*y + 27*y*y)
    return c1 * c2

def six_hump_camel(x, y):
    # Six-hump Camel
    return (4 - 2.1*x*x + (x**4)/3)*x*x + x*y + (-4 + 4*y*y)*y*y 