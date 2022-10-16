from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from sympy import factorint

optimal_inds = [625, 970, 314, 859, 697, 596, 807, 735, 775, 226, 294, 678, 254, 878, 706, 319,
                309, 212, 142, 356, 470, 713, 364, 381, 631, 645, 663, 518, 595, 949]

best_network = "oath_features_shufflenet_v2_x1_0_imagenet_torchvision.csv"


def open_figure(my_dpi=96, size=800, **kwargs):
    fig = plt.figure(figsize=(size / my_dpi, size / my_dpi), dpi=my_dpi, **kwargs)
    ax = fig.add_subplot(111)
    return fig, ax


def save_figure(fig: plt.figure, filename, size: tuple = (30, 30), dpi=600, **kwargs):
    cm = 1 / 2.54
    x = size[0] * cm
    y = size[1] * cm
    fig.set_size_inches(x * cm, y * cm)
    fig.savefig(filename + ".svg", dpi=dpi, **kwargs)
    fig.savefig(filename + ".png", dpi=dpi, **kwargs)
    fig.savefig(filename + ".pdf", dpi=dpi, **kwargs)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def determine_grid(n):
    s_primes = factorint(n)
    primes = []
    for key in s_primes.keys():
        for count in range(s_primes.get(key)):
            primes.append(key)
    primes = [prime for prime in permutations(primes)]
    primes = np.unique(primes, axis=0)
    l = primes.shape[1]
    diff = n
    id_a = None
    id_b = None
    for i in range(primes.shape[0]):
        for j in range(l):
            prime = primes[i, :]
            first_half = prime[:j]
            second_half = prime[j:]
            a = np.prod(first_half)
            b = np.prod(second_half)
            d = np.abs(a - b)
            if np.abs(a - b) < diff:
                id_a = a
                id_b = b
                diff = d
    return [id_a, id_b] if id_a < id_b else [id_b, id_a]