import jax
import jax.numpy as jnp
import jax.random as jrandom


def generate_voltera(
    n_steps: int,
    hurst: float = 0.1,
    xi: float = 0.235 * 0.235,
    eta: float = 1.0,
    dt: float = 1.0 / 250,
    return_dW: bool = False,
    *,
    rng_key: jrandom.PRNGKey
):
    """Simulate Volterra process from correlated 2D Brownian

    Implementation is based on this paper (Section 3.1)
    https://arxiv.org/pdf/1507.03004.pdf
    """

    a = hurst - 0.5
    # make covariance matrix
    cov = jnp.empty(shape=(2, 2))
    cov = cov.at[0, 0].set(dt)
    cov = cov.at[0, 1].set(dt ** (a + 1) / (a + 1))
    cov = cov.at[1, 0].set(cov[0, 1])
    cov = cov.at[1, 1].set(dt ** (2 * a + 1) / (2 * a + 1))

    # compute Cholesky decomposition for the covariance matrix
    chol = jnp.linalg.cholesky(cov)
    randn = jrandom.normal(shape=(n_steps, 2), key=rng_key)
    # dW is a 2D correlated noise (induced Brownian motion)
    dW1 = jax.vmap(lambda x: jnp.matmul(chol, x))(randn)

    # let take the second dimension
    Y1 = dW1[..., -1]
    Y1 = jnp.pad(Y1, (1, 0), constant_values=0.0)
    step = jnp.arange(1, n_steps + 1)
    b = step ** (a + 1)
    b = jnp.diff(b)
    b = b / (a + 1)
    b = b ** (1 / a)

    # the first dimension
    X = dW1[..., 0]
    X = jnp.pad(X, (1, 0), constant_values=0.0)
    G = jnp.zeros(shape=(1 + n_steps,))
    G = G.at[2:].set((b * dt) ** a)

    GX = jnp.convolve(G, X)
    Y2 = GX[: n_steps + 1]

    Y = jnp.sqrt(2 * a + 1) * (Y1 + Y2)
    t = dt * jnp.arange(n_steps + 1)
    V = xi * jnp.exp(eta * Y - 0.5 * eta**2 * t ** (2 * a + 1))
    V = V[..., :-1]

    if return_dW:
        return V, dW1

    return V
