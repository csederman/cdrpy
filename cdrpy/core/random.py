""""""

from __future__ import annotations

import random as rn
import numpy.random as np_rn

from collections import namedtuple

RandomState = namedtuple("RandomState", "random, numpy")


def _global_state():
    return RandomState(rn, np_rn)


def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))


_rs = _global_state()


def random_state(seed):
    """
    Creates a RandomState using the provided seed. If seed is None,
    return the global RandomState.

    Args:
        seed (int, optional): random seed

    Returns:
        RandomState object
    """
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)


def set_seed(seed):
    """
    Create a new global RandomState using the provided seed. If seed is None, StellarGraph's global
    RandomState object simply wraps the global random state for each external module.

    When trying to create a reproducible workflow using this function, please note that this seed
    only controls the randomness of the non-TensorFlow part of the library. Randomness within
    TensorFlow layers is controlled via TensorFlow's own global random seed, which can be set using
    ``tensorflow.random.set_seed``.

    Args:
        seed (int, optional): random seed

    """
    global _rs
    if seed is None:
        _rs = _global_state()
    else:
        _rs = _seeded_state(seed)
