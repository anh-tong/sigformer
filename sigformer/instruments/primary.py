from abc import abstractmethod
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.random import PRNGKey
from jaxtyping import Array, Float

from ..stochastic_process.forward_variance import generate_forward_variance
from ..stochastic_process.heston import generate_heston
from .base import BaseInstrument


class BasePrimary(BaseInstrument):

    dt: float
    cost: float

    @abstractmethod
    def simulate(self, rng_key: PRNGKey, n_steps: int, init_state=None):
        """Simulate primary"""


class BrownianStock(BasePrimary):
    def __init__(
        self,
        sigma: float = 0.2,
        mu: float = 0.2,
        cost: float = 0.0,
        dt: float = 1.0 / 250,
    ) -> None:

        self.sigma = sigma
        self.mu = mu
        self.cost = cost
        self.dt = dt

    def volatility(self, **kwargs):
        return self.sigma

    def variance(self, **kwargs):
        return self.sigma**2

    def simulate(
        self, rng_key: PRNGKey, n_steps: int, init_state=(0.0,)
    ) -> Dict[str, Float[Array, " n_steps"]]:

        S0 = init_state[0]
        randn = jrandom.normal(key=rng_key, shape=(n_steps,))
        randn = randn.at[0].set(0.0)
        drift = self.mu + self.dt + jnp.arange(n_steps)
        diffusion = np.sqrt(self.dt) + jnp.cumsum(randn)
        brown = drift + self.sigma * diffusion

        t = self.dt + jnp.arange(n_steps)
        spot = S0 * jnp.exp(brown - (self.sigma**2) * t / 2)

        return {
            "spot": spot,
        }


class RoughBergomiStock(BasePrimary):
    def __init__(
        self,
        hurst: float = 0.1,
        rho: float = -0.7,
        xi: float = 0.235**2,
        eta: float = 1.9,
        cost: float = 0.0,
        dt: float = 1.0 / 250,
        forward_offset: float = 15.0 / 250,
    ) -> None:
        super().__init__()

        # assert (hurst > 0) and (hurst < 1), "Hurst exponent should be in (0,1)"
        self.hurst = jnp.asarray(hurst)
        self.rho = jnp.asarray(rho)
        self.xi = jnp.asarray(xi)
        self.eta = jnp.asarray(eta)
        self.dt = jnp.asarray(dt)
        self.cost = jnp.asarray(cost)
        self.forward_offset = jnp.asarray(forward_offset)
        self.forward_offset_steps = jnp.ceil(self.forward_offset / self.dt)

        self.a = self.hurst - 0.5

        self.cholesky_cov = self._compute_chol_cov()

    def volatility(self, **kwargs):
        variance = kwargs["variance"]
        return jnp.sqrt(jnp.clip(variance, a_min=0.0))

    @property
    def default_init_state(self) -> Tuple[Float, Float, Float]:
        return (1.0, self.xi, None)

    def _compute_chol_cov(self) -> Float[Array, "2 2"]:
        """Compute Cholesky decomposition for covariance of 2D correlated Brownian noise
        It will be used to sample 2D correlated Brownian noise
        """
        # make covariance matrix
        cov = jnp.empty(shape=(2, 2))
        cov = cov.at[0, 0].set(self.dt)
        cov = cov.at[0, 1].set(self.dt ** (self.a + 1) / (self.a + 1))
        cov = cov.at[1, 0].set(cov[0, 1])
        cov = cov.at[1, 1].set(self.dt ** (2 * self.a + 1) / (2 * self.a + 1))

        # compute Cholesky decomposition for the covariance matrix
        chol = jnp.linalg.cholesky(cov)

        return chol

    def _sample_dW1(
        self,
        n_steps: int,
        rng_key: PRNGKey,
    ) -> Float[Array, "n_steps 2"]:
        """Sample 2D correlated Brownian noise"""
        randn = jrandom.normal(shape=(n_steps, 2), key=rng_key)
        return jax.vmap(lambda x: jnp.matmul(self.cholesky_cov, x))(randn)

    def _generate_price(
        self,
        theta: Float[Array, " n_steps"],
        dB: Float[Array, " n_steps+1"],
        n_steps: int,
        init_spot: float,
        init_variance: float,
    ) -> Tuple[Float[Array, " n_steps"], Float[Array, " n_steps"]]:

        # compute variance
        t = self.dt * jnp.arange(n_steps)
        variance = init_variance * jnp.exp(
            theta - 0.5 * self.eta**2 * t ** (2 * self.hurst)
        )

        # # compute forward variance Eq. 3.8
        # T_forward = (self.forward_offset_steps + n_steps) * self.dt
        # term2 = T_forward - jnp.arange(n_steps) * self.dt
        # term2 = term2 ** (2 * self.hurst) - T_forward ** (2 * self.hurst)
        # forward_variance = init_variance * jnp.exp(theta + 0.5 * self.eta**2 * term2)

        increments = (
            jnp.sqrt(variance[..., :-1]) * dB - 0.5 * variance[..., :-1] * self.dt
        )
        integral = jnp.cumsum(increments, axis=0)
        integral = jnp.pad(integral, (1, 0), constant_values=0.0)

        spot = init_spot * jnp.exp(integral)

        return spot, variance  # , forward_variance

    def simulate(
        self, rng_key: PRNGKey, n_steps: int, init_state=None
    ) -> Tuple[Float[Array, " n_steps"]]:

        dw1_key, dW2_key, fwd_var_key = jrandom.split(rng_key, 3)
        if init_state is None:
            init_state = self.default_init_state

        n_forward_steps = n_steps  # + self.forward_offset_steps
        s0, xi, directional = init_state

        # Sample 2D correlated Brownian noise
        dW1 = self._sample_dW1(n_steps=n_forward_steps, rng_key=dw1_key)
        # Sample 1D Brownian noise
        dW2 = jrandom.normal(shape=(n_forward_steps,), key=dW2_key) * np.sqrt(self.dt)

        # ----------------- Voltera process --------------------- #
        Y1 = dW1[..., -1]
        Y1 = jnp.pad(Y1, (1, 0), constant_values=0.0)

        gamma = jnp.arange(1, n_forward_steps + 1)
        gamma = gamma ** (self.a + 1)
        gamma = jnp.diff(gamma)
        gamma = gamma / (self.a + 1)
        gamma = gamma ** (1 / self.a)
        gamma = (gamma * self.dt) ** self.a
        gamma = jnp.pad(gamma, (2, 0), constant_values=0.0)

        X = dW1[..., 0]
        gamma_X_convolve = jnp.convolve(gamma, X)
        Y2 = gamma_X_convolve[: n_forward_steps + 1]

        Y_forward = jnp.sqrt(2 * self.hurst) * (Y1 + Y2)
        dB_forward = self.rho * dW1[..., 0] + jnp.sqrt(1 - self.rho**2) * dW2

        Y, dB = Y_forward[:n_steps], dB_forward[: n_steps - 1]
        theta = self.eta * Y

        if directional is not None:
            theta = directional

        spot, variance = self._generate_price(
            theta=theta,
            dB=dB,
            n_steps=n_steps,
            init_spot=s0,
            init_variance=xi,
        )

        forward_variance = generate_forward_variance(
            n_steps=n_steps,
            n_forward_steps=n_steps + self.forward_offset_steps,
            hurst=self.hurst,
            xi=self.xi,
            eta=self.eta,
            dt=self.dt,
            rng_key=fwd_var_key,
        )

        return {
            "spot": spot,
            "variance": variance,
            "forward_variance": forward_variance,
        }


class HestonStock(BasePrimary):
    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.2,
        rho: float = -0.7,
        cost: float = 0.0,
        dt: float = 1.0 / 250,
    ) -> None:
        super().__init__()

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.cost = cost
        self.dt = dt

    @property
    def default_init_state(self):
        return (1.0, self.theta)

    def volatility(self, **kwargs):
        variance = kwargs["variance"]
        return jnp.sqrt(jnp.clip(variance, a_min=0.0))

    def simulate(self, rng_key: PRNGKey, n_steps: int, init_state=None):

        if init_state is None:
            init_state = self.default_init_state

        spot, variance = generate_heston(
            rng_key=rng_key,
            n_steps=n_steps,
            init_state=init_state,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            dt=self.dt,
        )

        return {
            "spot": spot,
            "variance": variance,
        }
