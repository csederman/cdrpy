""""""

import typing as t


def is_real_iterable(x):
    """
    Tests if x is an iterable and is not a string.

    Parameters
    ----------
        x: a variable to check for whether it is an iterable

    Returns
    -------
        True if x is an iterable (but not a string) and False otherwise
    """
    return isinstance(x, t.Iterable) and not isinstance(x, (str, bytes))
