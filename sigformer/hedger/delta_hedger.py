from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, Float
from sigformer.hedger.base import BaseHedger
from sigformer.instruments.derivative import BaseDerivative
from sigformer.instruments.primary import HestonStock, RoughBergomiStock
from tqdm import tqdm


class HestonDeltaHedger(BaseHedger):

    hedge: List[BaseDerivative]
    underlier: HestonStock
    n_paths: int
    batch_size: int = 1000

    def __init__(self, derivative, hedge=None, n_paths=1000):
        self.derivative = derivative
        self.underlier = self.derivative.underlier
        assert isinstance(self.underlier, HestonStock)
        self.n_paths = n_paths

        if hedge is None:
            self.hedge = [self.derivative]
        else:
            self.hedge = hedge

    def compute_hedge(
        self, simulated_data, *, key
    ) -> Float[Array, "n_paths n_steps dim"]:

        spot, variance = simulated_data["spot"], simulated_data["variance"]
        n_paths = spot.shape[0]

        def compute_delta_for_a_point(spot, variance, remaining_steps, key):
            return self._delta(
                spot=spot,
                variance=variance,
                remaining_steps=remaining_steps,
                n_paths=self.n_paths,
                rng_key=key,
            )

        @partial(jax.jit, static_argnames="remaining_steps")
        def compute_delta_for_path(spot, variance, remaining_steps, key):
            n_paths = spot.shape[0]
            key = jrandom.split(key, n_paths)
            fn = partial(compute_delta_for_a_point, remaining_steps=remaining_steps)
            return jax.vmap(fn)(spot=spot, variance=variance, key=key)

        delta_prices, delta_variances = [], []
        for i in tqdm(range(spot.shape[-1] - 1)):
            n_batchs = n_paths // self.batch_size
            delta_price, delta_variance = [], []
            for index in range(n_batchs):
                selected = slice(index * self.batch_size, (index + 1) * self.batch_size)
                b_price, b_variance = compute_delta_for_path(
                    spot=spot[selected, i],
                    variance=variance[selected, i],
                    remaining_steps=spot.shape[-1] - i,
                    key=jrandom.fold_in(key, i),
                )
                delta_price.append(b_price)
                delta_variance.append(b_variance)
            delta_price = jnp.concatenate(delta_price, axis=0)
            delta_variance = jnp.concatenate(delta_variance)

            delta_prices.append(delta_price)
            delta_variances.append(delta_variance)

        delta_prices = jnp.stack(delta_prices, axis=-1)
        delta_variances = jnp.stack(delta_variances, axis=-1)
        # delta_variances = jnp.zeros_like(delta_prices)

        delta = jnp.stack([delta_prices, delta_variances], axis=-1)
        # do nothing at the maturity time by set delta = 0 at T
        delta = jnp.pad(delta, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)

        return delta

    def _delta(
        self,
        spot: float,
        variance: float,
        remaining_steps: int,
        n_paths: int,
        *,
        rng_key: jrandom.PRNGKey
    ):
        """
        Compute delta hedge using Eq. 5.6 in "Deep hedging" paper.
        """

        def _compute_payoff(s, v, key):
            key = jrandom.split(key, n_paths)

            output = jax.vmap(
                partial(
                    self.underlier.simulate, n_steps=remaining_steps, init_state=(s, v)
                )
            )(rng_key=key)
            spot = output["spot"]
            payoff = self.derivative.payoff(spot)
            return jnp.mean(payoff)

        du_ds, du_dv = jax.grad(_compute_payoff, argnums=(0, 1))(
            spot, variance, key=rng_key
        )

        # du_ds, du_dv = finite_diff(_compute_payoff,
        #                            [spot, variance, rng_key],
        #                            args_num=(0, 1),
        #                            epsilon=1e-4)

        # the denominator of delta hedge for variance can be computed explicitly
        alpha = self.underlier.kappa
        dL_dv = (1 - np.exp(-alpha * remaining_steps * self.underlier.dt)) / alpha

        return du_ds, du_dv - dL_dv

    def get_prices(self, simulated_data):
        """Get the prices of tradable assets"""
        prices = jnp.stack([h.price(simulated_data) for h in self.hedge], axis=-1)
        return prices


class RBergomiDeltaHedger(BaseHedger):

    underlier: RoughBergomiStock
    n_paths: int
    batch_size: int = 1000

    def __init__(self, derivative: BaseDerivative, n_paths=1000):
        self.derivative = derivative
        self.underlier = derivative.underlier
        assert isinstance(self.underlier, RoughBergomiStock)
        self.n_paths = n_paths

    def compute_hedge(
        self,
        simulated_data: Dict[str, Float[Array, "n_paths n_steps"]],
        *,
        key: jrandom.PRNGKey
    ) -> Float[Array, "n_paths n_steps dim"]:

        spot = simulated_data["spot"]
        variance = simulated_data["variance"]
        forward_variance = simulated_data["forward_variance"]

        forward_offset_steps = self.underlier.forward_offset_steps

        def compute_delta_for_a_point(
            spot, variance, forward_variance, remaining_steps_spot, key
        ):
            return self._delta(
                spot=spot,
                variance=variance,
                forward_variance=forward_variance,
                remaining_steps_spot=remaining_steps_spot,
                remaining_steps_variance=remaining_steps_spot + forward_offset_steps,
                n_paths=self.n_paths,
                rng_key=key,
            )

        @partial(jax.jit, static_argnames="remaining_steps_spot")
        def compute_delta_for_path(
            spot, variance, forward_variance, remaining_steps_spot, key
        ):
            n_paths = spot.shape[0]
            key = jrandom.split(key, n_paths)
            fn = partial(
                compute_delta_for_a_point, remaining_steps_spot=remaining_steps_spot
            )
            return jax.vmap(fn)(
                spot=spot, variance=variance, forward_variance=forward_variance, key=key
            )

        delta_prices, delta_variances = [], []
        for i in tqdm(range(spot.shape[-1] - 1)):
            delta_price, delta_variance = compute_delta_for_path(
                spot=spot[..., i],
                variance=variance[..., i],
                forward_variance=forward_variance[..., i:],
                remaining_steps_spot=spot.shape[-1] - i,
                key=jrandom.fold_in(key, i),
            )

            delta_prices.append(delta_price)
            delta_variances.append(delta_variance)

        delta_prices = jnp.stack(delta_prices, axis=-1)
        delta_variances = jnp.stack(delta_variances, axis=-1)

        delta = jnp.stack([delta_prices, delta_variances], axis=-1)
        delta = jnp.pad(delta, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)

        return delta

    def _delta(
        self,
        spot: Float,
        variance: Float,
        forward_variance: Float[Array, " remaining_steps"],
        remaining_steps_spot: int,
        remaining_steps_variance: int,
        n_paths: int,
        *,
        rng_key: jrandom.PRNGKey
    ) -> Tuple[Float, Float]:
        """
        This follows Sect. 3.2 of "Deep hedging under Rough Volatity" paper.

        However, the most detailed description is in
        "A Martingale Approach for Fractional Brownian Motions and Related Path Dependent PDEs"

        """
        hurst = self.underlier.hurst

        # `a` is the directional for Gateaux derivative
        a = remaining_steps_variance - jnp.arange(0, remaining_steps_spot)
        a = a * self.underlier.dt
        a = a ** (hurst - 0.5)

        # obtain \\Theta_t back from `forward_variance`
        T_forward = (
            self.derivative.n_steps + self.underlier.forward_offset_steps
        ) * self.underlier.dt
        t = self.derivative.maturity - remaining_steps_spot * self.underlier.dt
        term = (T_forward - t) ** (2 * hurst) - T_forward ** (2 * hurst)
        theta = (
            jnp.log(forward_variance / variance) - 0.5 * self.underlier.eta**2 * term
        )

        def _compute_payoff(s0, eps, key):
            theta_eps = eps * a + theta
            output = self.underlier.simulate(
                rng_key=key,
                n_steps=remaining_steps_spot,
                init_state=(s0, variance, theta_eps),
            )
            spot = output["spot"]
            # compute payoff
            payoff = self.derivative.payoff(spot=spot)

            return payoff

        def _fun(s0, eps, key):
            """return mean of payoff"""
            key = jrandom.split(key, n_paths)
            payoff = jax.vmap(lambda k: _compute_payoff(s0, eps, k))(key)
            return jnp.mean(payoff)

        # compute two derivatives like in Eq. 3.7
        # note that we approximate Gateaux derivatives by sending \\epsilon to 0
        # also, we use auto-diff instead of finite diference
        grad = jax.grad(_fun, argnums=(0, 1))(spot, jnp.asarray(0.0), key=rng_key)
        delta_price, directional_grad = grad

        # this is the second term in Eq. 3.7
        time_to_maturity = remaining_steps_variance * self.underlier.dt
        delta_variance = (
            time_to_maturity ** (-self.underlier.hurst + 0.5)
            * directional_grad
            / forward_variance[0]
        )

        return delta_price, delta_variance

    def get_prices(self, simulated_data: Dict[str, Array]) -> Array:
        spot = simulated_data["spot"]
        forward_variance = simulated_data["forward_variance"]
        return jnp.stack([spot, forward_variance], axis=-1)


if __name__ == "__main__":

    from sigformer.instruments.derivative import EuropeanOption

    # rough Bergomi
    stock = RoughBergomiStock()
    derivative = EuropeanOption(stock)

    delta_hedger = RBergomiDeltaHedger(derivative)
    delta_hedger.compute_pnl(
        rng_key=jrandom.PRNGKey(0),
        n_paths=1000,
    )

    # Heston
    stock = HestonStock()
    derivative = EuropeanOption(stock)
    delta_hedger = HestonDeltaHedger(derivative)

    spot, variance = jnp.asarray(1.0), jnp.asarray(0.2)

    ds, dv = delta_hedger._delta(
        spot=spot,
        variance=variance,
        remaining_steps=10,
        n_paths=10000,
        rng_key=jrandom.PRNGKey(0),
    )

    delta_hedger.compute_pnl(
        rng_key=jrandom.PRNGKey(0),
        n_paths=1000,
    )
