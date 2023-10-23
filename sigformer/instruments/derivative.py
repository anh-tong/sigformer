from abc import abstractmethod
from math import ceil
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array

from ..utils import realized_variance
from .base import BaseInstrument
from .primary import BasePrimary


class BaseDerivative(BaseInstrument):

    underlier: BasePrimary
    cost: float
    maturity: float
    n_steps: int
    pricer: Optional[Callable[[Any], Array]]

    def __init__(self) -> None:
        self.pricer = None
        self.cost = 0.0

    def price(self, input: Dict[str, Array]):
        spot = input["spot"]
        if self.pricer is not None:
            return self.pricer(input)
        return spot

    @abstractmethod
    def payoff(self, spot=None, rng_key: PRNGKey = None, init_state=None):
        """Compute payoff"""

    def simulate(self, rng_key: PRNGKey, init_state):
        return self.underlier.simulate(
            rng_key=rng_key, n_steps=self.n_steps, init_state=init_state
        )


class Mixin:

    underlier: BaseDerivative
    strike: float
    maturity: float

    def moneyness(
        self,
        time_step: Optional[int] = None,
        log: bool = False,
        spot: Optional[jnp.array] = None,
        init_state: Optional[Any] = None,
        rng_key: PRNGKey = None,
    ):

        index = ... if time_step is None else time_step
        if spot is None:
            assert rng_key is not None
            assert init_state is not None
            spot = self.underlier.simulate(
                rng_key=rng_key, time_horizon=self.maturity, init_state=init_state
            )

        output = spot[index] / self.strike
        if log:
            return jnp.log(output)
        return output

    def log_moneyness(
        self,
        time_step: Optional[int] = None,
        spot: Optional[jnp.ndarray] = None,
        init_state: Optional[Any] = None,
        rng_key: Optional[PRNGKey] = None,
    ):
        return self.moneyness(
            time_step, log=True, spot=spot, init_state=init_state, rng_key=rng_key
        )

    def time_to_maturity(self, time_step: Optional[int] = None):

        n_steps = ceil(self.maturity / self.underlier.dt + 1)
        if time_step is None:
            t = jnp.arange(n_steps) * self.underlier.dt
            return t[-1] - t
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = jnp.array([time]) * self.underlier.dt
            return t

    def max_moneyness(
        self,
        time_step: Optional[int] = None,
        log: bool = False,
        spot: Optional[jnp.array] = None,
        init_state: Optional[Any] = None,
        rng_key: Optional[PRNGKey] = None,
    ):

        moneyness = self.moneyness(None, log, spot, init_state, rng_key)
        if time_step is None:
            return jax.lax.cummax(moneyness, axis=0)
        else:
            return jnp.max(moneyness[: (time_step + 1)], axis=0, keepdims=True)

    def max_log_moneyness(
        self,
        time_step=None,
        spot=None,
        init_state=None,
        rng_key=None,
    ):
        return self.max_moneyness(
            time_step=time_step,
            log=True,
            spot=spot,
            init_state=init_state,
            rng_key=rng_key,
        )


class EuropeanOption(BaseDerivative, Mixin):
    def __init__(
        self,
        underlier: BaseDerivative,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20.0 / 250,
    ) -> None:
        super().__init__()
        self.call = call
        self.strike = strike
        self.maturity = maturity
        self.underlier = underlier
        self.n_steps = ceil(self.maturity / self.underlier.dt)

    def payoff(self, spot=None, rng_key: PRNGKey = None, init_state=None):
        if spot is None:
            # need to simulate the underlier
            assert rng_key is not None
            assert init_state is not None
            output = self.simulate(rng_key=rng_key, init_state=init_state)
            spot = output["spot"]

        if self.call:
            return jax.nn.relu(spot[..., -1] - self.strike)
        else:
            return jax.nn.relu(self.strike - spot[..., -1])


class VarianceSwap(BaseDerivative):
    def __init__(
        self,
        underlier: BasePrimary,
        maturity: float = 20.0 / 250,
        strike: float = 0.04,
        pricer: Callable = None,
    ) -> None:
        super().__init__()
        self.underlier = underlier
        self.maturity = maturity
        self.strike = strike
        self.pricer = pricer

    def payoff(self, spot=None, rng_key: PRNGKey = None, init_state=None):

        if spot is None:
            spot = self.underlier.simulate(
                rng_key,
                n_steps=ceil(self.maturity / self.underlier.dt),
                init_state=init_state,
            )

        return realized_variance(spot, dt=self.underlier.dt) - self.strike
