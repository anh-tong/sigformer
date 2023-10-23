import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
from sigformer.stochastic_process.cir import generate_cir


def generate_heston(
    rng_key: jrandom.PRNGKey,
    n_steps: int,
    init_state=None,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 0.2,
    rho: float = -0.7,
    dt: float = 1.0 / 250,
) -> Float[Array, " n_steps"]:
    """
    Generate Heston process
        dS(t) = S(t) sqrt(V(t)) dW1(t)
        dV(t) = kappa (theta - V(t)) dt + sigma sqrt(V(t)) dW2(t)
    """

    if init_state is None:
        init_state = (1.0, theta)

    variance_key, spot_key = jrandom.split(rng_key)

    variance = generate_cir(
        rng_key=variance_key,
        n_steps=n_steps,
        init_state=init_state[1:],
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
    )
    randn = jrandom.normal(key=spot_key, shape=(n_steps - 1,))

    GAMMA1, GAMMA2 = 0.5, 0.5

    def _body_fn(carry, i):

        k0 = -rho * kappa * theta * dt / sigma
        k1 = GAMMA1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = GAMMA2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = GAMMA1 * dt * (1 - rho**2)
        k4 = GAMMA2 * dt * (1 - rho**2)
        v0 = variance[i]
        v1 = variance[i + 1]
        output = carry + k0 + k1 * v0 + k2 * v1 + jnp.sqrt(k3 * v0 + k4 * v1) * randn[i]

        return output, output

    init = jnp.log(jnp.asarray(init_state[0]))
    _, log_spot = jax.lax.scan(f=_body_fn, init=init, xs=jnp.arange(n_steps - 1))

    log_spot = jnp.concatenate([init[None], log_spot])

    return jnp.exp(log_spot), variance


if __name__ == "__main__":

    rng_key = jrandom.PRNGKey(0)
    n_steps = 100
    spot, variance = generate_heston(rng_key, n_steps)
    print(spot.shape)
    print(variance.shape)
