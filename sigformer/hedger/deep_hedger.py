from typing import Callable, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float

from ..feature import FeatureList
from ..instruments.base import BaseInstrument
from ..instruments.derivative import BaseDerivative
from ..loss import BaseCriterion, EntropyRiskMeasure
from .base import BaseHedger


class DeepHedger(BaseHedger):

    model: eqx.Module
    inputs: FeatureList = eqx.static_field()
    criterion: BaseCriterion = eqx.static_field()
    derivative: BaseDerivative = eqx.static_field()
    hedge: List[BaseInstrument] = eqx.static_field()

    recur_type: str = eqx.static_field()
    n_inputs: int = eqx.static_field()
    n_outputs: int = eqx.static_field()
    n_recur_hidden: int = eqx.static_field()
    model_width: int = eqx.static_field()
    model_depth: int = eqx.static_field()

    def __init__(
        self,
        derivative: BaseDerivative,
        inputs: List[str],
        hedge=None,
        criterion=EntropyRiskMeasure(),
        recur_type="SemiRecur",  # or NoRecur
        model_width=128,
        model_depth=5,
        *,
        rng_key: jrandom.PRNGKey,
    ) -> None:
        self.inputs = FeatureList(feature=inputs, derivative=derivative)
        self.derivative = derivative
        self.criterion = criterion
        if hedge is None:
            self.hedge = [derivative]
        else:
            self.hedge = hedge

        self.recur_type = recur_type
        # let's just make the number of recurrent hidden equal to the number of output in the case of fully recurence
        self.n_recur_hidden = len(self.hedge) if recur_type == "Recur" else 0
        self.n_outputs = len(self.hedge) + self.n_recur_hidden
        if recur_type == "NoRecur":
            additional_n_inputs = 0
        elif recur_type == "SemiRecur":
            additional_n_inputs = self.n_outputs
        elif recur_type == "Recur":
            additional_n_inputs = self.n_recur_hidden
        else:
            raise ValueError(f"Unrecognized `recur_type`: {recur_type}")
        self.n_inputs = len(inputs) + additional_n_inputs
        self.model_width = model_width
        self.model_depth = model_depth
        self.model = self.initialize_model(rng_key)

    def initialize_model(
        self, rng_key: jrandom.PRNGKey
    ) -> Callable[[Float[Array, " n_steps"]], Float[Array, " "]]:
        model = eqx.nn.MLP(
            in_size=self.n_inputs,
            out_size=self.n_outputs,
            width_size=self.model_width,
            depth=self.model_depth,
            key=rng_key,
        )
        return model

    def compute_hedge(
        self,
        simulated_data: Dict[str, Float[Array, "n_paths n_steps"]],
        *,
        key: jrandom.PRNGKey = None,
    ):
        """
        How to get the previous hedging value
        """
        n_steps = simulated_data["spot"].shape[-1]

        def _compute_hedge(single_path: Dict[str, Float[Array, " n_steps"]]):

            if self.recur_type == "SemiRecur":

                def scan_fn(carry, i):
                    input = self.inputs.get(single_path, i)
                    # no gradient for `carry`. otherwise, this will be a fully recurrent model
                    carry = jax.lax.stop_gradient(carry)
                    input = jnp.concatenate([input, carry], axis=0)
                    output = self.model(input)
                    return output, output

                init = jnp.zeros(shape=(self.n_outputs,))
                _, output = jax.lax.scan(f=scan_fn, init=init, xs=jnp.arange(n_steps))
                return output
            elif self.recur_type == "Recur":

                def scan_fn(carry, i):
                    input = self.inputs.get(single_path, i)
                    input = jnp.concatenate([input, carry], axis=0)
                    output = self.model(input)
                    carry, output = jnp.split(
                        output, indices_or_sections=(self.n_recur_hidden,)
                    )
                    return carry, output

                init = jnp.zeros(shape=(self.n_recur_hidden,))
                _, output = jax.lax.scan(f=scan_fn, init=init, xs=jnp.arange(n_steps))
                return output
            else:
                input = self.inputs.get(single_path)
                # extract features from spot price
                output = jax.vmap(self.model)(input)
                return output

        return jax.vmap(_compute_hedge)(simulated_data)

    def __call__(
        self, input: Float[Array, "n_steps n_features"]
    ) -> Float[Array, "n_steps n_hedges"]:
        return self.model(input)

    def get_prices(
        self, simulated_data: Dict[str, Float[Array, "n_paths n_steps"]]
    ) -> Float[Array, "n_path n_steps dim"]:
        """Get the prices of tradable assets"""
        prices = jnp.stack([h.price(simulated_data) for h in self.hedge], axis=-1)
        return prices

    def compute_loss(
        self,
        init_state,
        n_paths=1000,
        n_times=1,
        *,
        rng_key: jrandom.PRNGKey,
    ) -> Float:
        def _portfolio_and_payoff(key):
            return self.compute_pnl(
                rng_key=key,
                n_paths=n_paths,
                init_state=init_state,
                return_portfolio_and_payoff=True,
            )

        def _compute_loss(key):
            portfolio, payoff = _portfolio_and_payoff(key)
            return self.criterion(portfolio, payoff)

        if n_times == 1:
            return _compute_loss(rng_key)
        else:
            batch_compute_loss = jax.vmap(_compute_loss)
            rng_key = jrandom.split(rng_key, n_times)
            return jnp.mean(batch_compute_loss(rng_key), axis=0)
