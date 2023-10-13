# %%
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def staircase(x, n, d, scale=1000):
    offsets = np.linspace(1/n, 1, n)
    stairs = np.zeros_like(x)
    for i in range(d):
        saw = x - stairs
        baby_stairs = sigmoid(( saw[:, None] - offsets) * scale).sum(-1) / n
        stairs = baby_stairs / n ** i + stairs
        offsets /= n
    return stairs

x = np.linspace(0, 1, 1000)
n = 3
for d in [1, 2, 3, 4]:
    plt.plot(x, staircase(x, n, d), label=f"n={n}, depth={d}")
plt.legend()