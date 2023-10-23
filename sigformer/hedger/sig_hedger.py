from typing import Callable, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
from sigformer.feature import FeatureList
from sigformer.hedger.base import BaseHedger
from sigformer.instruments.base import BaseInstrument
from sigformer.instruments.derivative import BaseDerivative
from sigformer.loss import BaseCriterion, EntropyRiskMeasure
from sigformer.nn.model import (
    Config,
    SigFormer,
    SigFormer_v2,
    Signature,
    TensorFlatten,
    VallinaTransformer,
)


class SigHedger(BaseHedger):

    model: eqx.Module
    config: Config = eqx.static_field()
    inputs: FeatureList = eqx.static_field()
    criterion: BaseCriterion = eqx.static_field()
    derivative: BaseDerivative = eqx.static_field()
    hedge: List[BaseInstrument] = eqx.static_field()

    n_inputs: int = eqx.static_field()
    n_outputs: int = eqx.static_field()

    def __init__(
        self,
        derivative,
        inputs,
        hedge=None,
        criterion=EntropyRiskMeasure(),
        signature_depth=3,
        model_dim=2,
        n_attn_heads=1,
        n_attn_blocks=2,
        d_ff=12,
        dropout=0.1,
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

        self.n_inputs = len(inputs)
        self.n_outputs = len(self.hedge)
        self.config = Config(
            in_dim=self.n_inputs,
            out_dim=self.n_outputs,
            dim=model_dim,
            num_heads=n_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_attn_blocks=n_attn_blocks,
            order=signature_depth,
        )
        self.model = self.initialize_model(rng_key=rng_key)

    def initialize_model(self, rng_key: jrandom.PRNGKey) -> Callable[[Array], Array]:
        model = SigFormer(config=self.config, key=rng_key)
        return model

    def compute_hedge(
        self, simulated_data: Dict[str, Array], *, key: jrandom.PRNGKey = None
    ):
        def _compute_hedge(single_path):
            input = self.inputs.get(single_path)
            output = self.model(input, key=key)

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
        p0=None,
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
            payoff = payoff if p0 is None else p0
            return self.criterion(portfolio, payoff)

        if n_times == 1:
            return _compute_loss(rng_key)
        else:
            batch_compute_loss = jax.vmap(_compute_loss)
            rng_key = jrandom.split(rng_key, n_times)
            return jnp.mean(batch_compute_loss(rng_key), axis=0)


class SigHedger_v2(SigHedger):

    model: eqx.Module
    config: Config = eqx.static_field()
    inputs: FeatureList = eqx.static_field()
    criterion: BaseCriterion = eqx.static_field()
    derivative: BaseDerivative = eqx.static_field()
    hedge: List[BaseInstrument] = eqx.static_field()

    n_inputs: int = eqx.static_field()
    n_outputs: int = eqx.static_field()

    def __init__(
        self,
        derivative,
        inputs,
        hedge=None,
        criterion=EntropyRiskMeasure(),
        signature_depth=3,
        model_dim=2,
        n_attn_heads=1,
        n_attn_blocks=2,
        d_ff=12,
        dropout=0.1,
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

        self.n_inputs = len(inputs)
        self.n_outputs = len(self.hedge)
        self.config = Config(
            in_dim=self.n_inputs,
            out_dim=self.n_outputs,
            dim=model_dim,
            num_heads=n_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_attn_blocks=n_attn_blocks,
            order=signature_depth,
        )
        self.model = self.initialize_model(rng_key=rng_key)

    def initialize_model(self, rng_key: jrandom.PRNGKey) -> Callable[[Array], Array]:
        model = SigFormer_v2(config=self.config, key=rng_key)
        return model


class SignatureOnlyHedger(SigHedger):
    def __init__(
        self,
        derivative,
        inputs,
        hedge=None,
        criterion=EntropyRiskMeasure(),
        signature_depth=3,
        model_dim=-1,
        n_attn_heads=-1,
        n_attn_blocks=-1,
        d_ff=-1,
        dropout=-1,
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

        self.n_inputs = len(inputs)
        self.n_outputs = len(self.hedge)
        self.config = Config(
            in_dim=self.n_inputs,
            out_dim=self.n_outputs,
            dim=model_dim,
            num_heads=n_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_attn_blocks=n_attn_blocks,
            order=signature_depth,
        )
        self.model = self.initialize_model(rng_key=rng_key)

    def initialize_model(self, rng_key: jrandom.PRNGKey) -> Callable[[Array], Array]:

        sig_depth = self.config.order
        linear_input_dim = int(
            (self.n_inputs ** (sig_depth + 1) - 1) / (self.n_inputs - 1) - 1
        )
        linear_output_dim = self.n_outputs

        class SignatureLayer(eqx.Module):
            signature_fn: Signature
            flatten_fn: TensorFlatten
            linear: eqx.nn.Linear

            def __init__(self):
                self.signature_fn = Signature(depth=sig_depth)
                self.flatten_fn = TensorFlatten()
                self.linear = eqx.nn.Linear(
                    linear_input_dim, linear_output_dim, key=rng_key
                )

            def __call__(self, x, key=None):
                x = self.signature_fn(x)
                x = self.flatten_fn(x)
                return jax.vmap(self.linear)(x)

        return SignatureLayer()


class TransformerHedger(SigHedger):
    def __init__(
        self,
        derivative,
        inputs,
        hedge=None,
        criterion=EntropyRiskMeasure(),
        signature_depth=3,
        model_dim=3,
        n_attn_heads=12,
        n_attn_blocks=5,
        d_ff=12,
        dropout=0.1,
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

        self.n_inputs = len(inputs)
        self.n_outputs = len(self.hedge)
        self.config = Config(
            in_dim=self.n_inputs,
            out_dim=self.n_outputs,
            dim=model_dim,
            num_heads=n_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_attn_blocks=n_attn_blocks,
            order=signature_depth,
        )
        self.model = self.initialize_model(rng_key=rng_key)

    def initialize_model(self, rng_key: jrandom.PRNGKey) -> Callable[[Array], Array]:
        model = VallinaTransformer(config=self.config, key=rng_key)
        return model
