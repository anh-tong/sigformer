import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from .voltera import generate_voltera


def generate_rough_bergomi(
    n_steps: int,
    s0: float = 1.0,
    rho: float = -0.7,
    hurst: float = 0.1,
    eta: float = 1.9,
    xi: float = 0.235 * 0.235,
    dt: float = 1.0 / 250,
    *,
    rng_key: jrandom.PRNGKey
):
    key, voltera_key = jrandom.split(rng_key)
    V, dW1 = generate_voltera(
        n_steps=n_steps,
        hurst=hurst,
        xi=xi,
        eta=eta,
        dt=dt,
        return_dW=True,
        rng_key=voltera_key,
    )

    dW2 = jrandom.normal(shape=(n_steps,), key=key) * np.sqrt(dt)
    dB = rho * dW1[..., 0] + dW2 * jnp.sqrt(1 - rho**2)

    increments = jnp.sqrt(V) * dB - 0.5 * V * dt
    integral = jnp.cumsum(increments, axis=0)
    integral = jnp.pad(integral, (1, 0), constant_values=0.0)

    S = s0 * jnp.exp(integral)

    return S, V
