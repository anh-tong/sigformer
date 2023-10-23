import jax.random as jrandom


def split_key(key):
    return None if key is None else jrandom.split(key, 1)[0]
