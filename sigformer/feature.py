from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .instruments.derivative import BaseDerivative


class Feature(ABC):

    name: str
    derivative: BaseDerivative

    @abstractmethod
    def get(self, output: Dict[str, Array], time_step: Optional[int] = None):
        """Get feature at `time_step`"""


class Moneyness(Feature):

    log: bool

    def __init__(self, derivative: BaseDerivative, log: bool = False) -> None:
        super().__init__()
        self.name = "moneyness"
        self.derivative = derivative
        self.log = log

    def get(self, output: Dict[str, Array], time_step: Optional[int] = None):
        spot = output["spot"]
        return self.derivative.moneyness(time_step=time_step, spot=spot)


class LogMoneyness(Moneyness):
    def __init__(self, derivative: BaseDerivative) -> None:
        super().__init__(derivative=derivative, log=True)


class TimeToMaturity(Feature):
    def __init__(self, derivative: BaseDerivative) -> None:
        super().__init__()
        self.name = "time_to_maturity"
        self.derivative = derivative

    def get(self, output: Dict[str, Array], time_step: Optional[int], spot):
        spot = output["spot"]
        return self.derivative.time_to_maturity(time_step=time_step, spot=spot)


class Spot(Feature):

    log: bool

    def __init__(self, derivative, log: bool = False) -> None:
        super().__init__()
        self.name = "underlier_spot"
        self.derivative = None
        self.log = log

    def get(self, output: Dict[str, Array], time_step: Optional[int] = None):
        spot = output["spot"]
        if time_step is not None:
            output = spot[time_step]
        else:
            output = spot

        if self.log:
            output = jnp.log(output)

        return output


class LogSpot(Spot):
    def __init__(self, derivative) -> None:
        super().__init__(derivative, log=True)


class Volatility(Feature):
    def __init__(self, derivative: BaseDerivative):

        self.name = "volatility"
        self.derivative = derivative

    def get(self, output: Dict[str, Array], time_step: Optional[int] = None):
        vol = self.derivative.underlier.volatility(**output)
        if time_step is not None:
            return vol[time_step]
        else:
            return vol


FEATURES = [Moneyness, LogMoneyness, TimeToMaturity, Spot, LogSpot, Volatility]

FEATURES_DICT = dict((cls.__name__, cls) for cls in FEATURES)


class FeatureList(Feature):
    def __init__(self, feature: List[str], derivative: BaseDerivative) -> None:
        self.features = []
        for name in feature:
            if FEATURES_DICT.__contains__(name):
                f = FEATURES_DICT[name](derivative)
                self.features.append(f)
            else:
                raise ValueError(f"Unknown feature name: {name}")

    def get(
        self, spot, time_step: Optional[int] = None
    ) -> Float[Array, "n_paths n_steps n_features"]:
        output = [f.get(spot, time_step) for f in self.features]
        output = jnp.stack(output, axis=-1)
        return output
