import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import matplotlib.pyplot as plt


def windowed_view(x, window_size):
    """Creat a 2d windowed view of a 1d array.

    `x` must be a 1d numpy array.

    `numpy.lib.stride_tricks.as_strided` is used to create the view.
    The data is not copied.

    Example:

    > x = np.array([1, 2, 3, 4, 5, 6])
    > windowed_view(x, 3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])
    """
    y = as_strided(x, shape=(x.size - window_size + 1, window_size),
                   strides=(x.strides[0], x.strides[0]))
    return y


def rolling_max_dd(x, window_size, min_periods=1):
    """Compute the rolling maximum drawdown of `x`.

    `x` must be a 1d numpy array.
    `min_periods` should satisfy `1 <= min_periods <= window_size`.

    Returns an 1d array with length `len(x) - min_periods + 1`.
    """
    if min_periods < window_size:
        pad = np.empty(window_size - min_periods)
        pad.fill(x[0])
        x = np.concatenate((pad, x))
    y = windowed_view(x, window_size)
    running_max_y = np.maximum.accumulate(y, axis=1)
    dd = y - running_max_y
    return dd.min(axis=1)


def max_dd(ser):
    max2here = ser.cummax()
    dd2here = ser - max2here
    return dd2here.min()


if __name__ == '__main__':

    np.random.seed(0)
    n = 100
    window_length = 10

    s = pd.Series(np.random.randn(n).cumsum())
    df = pd.DataFrame(s)
    df.columns = ['s']

    rmdd = rolling_max_dd(s.values, window_length, min_periods=1)
    assert rmdd.shape == s.values.shape, "Shape mismatch"
    df.plot(linewidth=3, alpha=0.4)
    plt.plot(rmdd, 'g-', label='rolling MDD')
    plt.legend()
    plt.show()
