import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, Float


def generate_cir(
    rng_key: jrandom.PRNGKey,
    n_steps: int,
    init_state=None,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 0.2,
    dt: float = 1.0 / 250,
) -> Float[Array, " n_steps"]:
    """
    Generate Cox-Ingersoll-Ross process
        dX(t) = kappa (theta - X(t)) dt + sigma sqrt(X(t)) dW(t)

    Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic Volatility Model (January 23, 2007).
    Available at SSRN:
        https://ssrn.com/abstract=946405
    """

    # the condition that the process will never touch zero
    assert 2 * kappa * theta >= sigma**2

    if init_state is None:
        init_state = (theta,)

    PSI_CRIT = 1.5
    exp = np.exp(-kappa * dt)

    def _body_fn(carry, randn):
        EPSILON = jnp.finfo(carry.dtype).tiny
        m = theta + (carry - theta) * exp
        s2 = carry * (sigma**2) * exp * (1 - exp) / kappa
        s2 = s2 + theta * (sigma**2) * np.square(1 - exp) / (2 * kappa)
        psi = s2 / jnp.clip(jnp.square(m), a_min=EPSILON)

        # compute V(t + dt) where psi <= PSI_CRIT
        b = jnp.sqrt((2 / psi) - 1 + jnp.sqrt(2 / psi) * jnp.sqrt(2 / psi - 1))
        a = m / (1 + jnp.square(b))
        next_0 = a * jnp.square(b + randn[0])

        # compute V(t + dt) where psi > PSI_CRIT
        u = randn[1]
        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / jnp.clip(m, a_min=EPSILON)
        pinv = jnp.log((1 - p) / jnp.clip(1 - u, a_min=EPSILON)) / beta
        next_1 = jnp.where(u > p, pinv, jnp.zeros_like(u))

        # based on the considition, select the next value
        output = jnp.where(psi <= PSI_CRIT, next_0, next_1)

        return output, output

    randn = jrandom.normal(key=rng_key, shape=(n_steps - 1, 2))

    _, output = jax.lax.scan(f=_body_fn, init=jnp.asarray(init_state[0]), xs=randn)
    output = jnp.pad(output, (1, 0), constant_values=init_state[0])

    return output


if __name__ == "__main__":

    rng_key = jrandom.PRNGKey(0)
    n_steps = 100

    print(generate_cir(rng_key, n_steps).shape)
