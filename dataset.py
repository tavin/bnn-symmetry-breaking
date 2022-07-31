from collections import namedtuple

from jax import numpy as jnp
from numpyro.examples.datasets import load_dataset
from sklearn.model_selection import train_test_split


Data = namedtuple('Data', 'x_train x_test y_train y_test')
Split = namedtuple('Split', 'train_size random_state')


def load(dataset, split: Split, *, verbose=False) -> Data:
    """ Fetch and split any numpyro example dataset. """
    _, fetch = load_dataset(dataset, shuffle=False)
    x, y = fetch()
    x_train, x_test, y_train, y_test = train_test_split(x, y, **split._asdict())
    data = Data(*map(jnp.array, (x_train, x_test, y_train, y_test)))
    if verbose:
        for k, v in data._asdict().items():
            print(k + ':', v.shape)
    return data


def normalize(val, mean=None, std=None):
    """ Normalize data to zero mean, unit variance. """
    if mean is None and std is None:
        # Only use training data to estimate mean and std.
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std
