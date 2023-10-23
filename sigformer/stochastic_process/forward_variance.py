import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def generate_forward_variance(
    n_steps: int,
    n_forward_steps: int,
    hurst: float = 0.1,
    xi: float = 0.235**2,
    eta: float = 1.0,
    dt: float = 1.0 / 250,
    *,
    rng_key: jrandom.PRNGKey,
):

    brownian_noise = jrandom.normal(key=rng_key, shape=(n_steps - 1,)) * np.sqrt(dt)

    def scan_fn(carry, i):
        delta_T = (n_forward_steps - i) * dt
        ret = (
            carry
            + carry
            * jnp.sqrt(2 * hurst)
            * eta
            * delta_T ** (hurst - 0.5)
            * brownian_noise[i]
        )
        return ret, ret

    init = jnp.asarray(xi * np.exp(xi))
    _, forward_variance = jax.lax.scan(f=scan_fn, init=init, xs=jnp.arange(n_steps - 1))

    return jnp.concatenate([init[None], forward_variance])


if __name__ == "__main__":

    output = generate_forward_variance(10, 20, rng_key=jrandom.PRNGKey(0))
