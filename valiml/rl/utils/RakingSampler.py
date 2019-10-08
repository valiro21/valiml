import numpy as np


def harmonic_series_sum(n):
    return np.log(n) + 0.577215664


def harmonic_sampler(n_items):
    smallest_2_power = 2**int(np.log2(n_items) + 1)

    cutoff = np.random.random() * harmonic_series_sum(n_items)
    position = 0
    step = smallest_2_power
    while step > 0:
        position += step
        if position >= n_items or harmonic_series_sum(position + 1) > cutoff:
            position -= step
        step = int(step // 2)
    return position
