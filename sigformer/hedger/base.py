from functools import partial
from typing import Dict

import equinox as eqx
import jax
import jax.random as jrandom
from jaxtyping import Array, Float
from sigformer.instruments.derivative import BaseDerivative
from sigformer.utils import pl


class BaseHedger(eqx.Module):

    derivative: BaseDerivative

    def compute_hedge(
        self,
        simulated_data: Dict[str, Float[Array, "n_paths n_steps"]],
        *,
        key: "jax.random.PRNGKey"
    ) -> Float[Array, "n_paths dim n_steps"]:
        """Given `simulated_data`, this returns how many units should trade"""
        raise NotImplementedError

    def get_prices(
        self, simulated_data: Dict[str, Float[Array, "n_paths n_steps"]]
    ) -> Float[Array, "n_paths dim n_steps"]:
        """Get prices of tradable assets"""
        raise NotImplementedError

    def compute_pnl(
        self,
        rng_key: jrandom.PRNGKey,
        n_paths: int,
        simulated_data: Dict[str, Float[Array, "n_paths n_steps"]] = None,
        init_state=None,
        return_portfolio_and_payoff=False,
    ):
        """Compute Profit-N-Loss"""

        if rng_key is None:
            hedge_key = None
        else:
            rng_key, hedge_key = jrandom.split(rng_key)
        if simulated_data is None:
            rng_key = jrandom.split(rng_key, n_paths)
            simulated_data = jax.vmap(
                partial(self.derivative.simulate, init_state=init_state)
            )(rng_key)
        prices = self.get_prices(simulated_data)
        unit = self.compute_hedge(simulated_data, key=hedge_key)
        payoff = self.derivative.payoff(simulated_data["spot"])

        if return_portfolio_and_payoff:
            portfolio = jax.vmap(pl)(spot=prices, unit=unit)
            return portfolio, payoff
        else:
            return jax.vmap(pl)(spot=prices, unit=unit, payoff=payoff)
